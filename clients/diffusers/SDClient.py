import os
import numpy as np
import torch
from cv2 import Canny
from diffusers import (
    StableVideoDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderTiny,
)
from settings import (
    DEVICE,
    SD_MODEL,
    SD_USE_HYPERTILE,
    SD_USE_VAE,
    SD_USE_SDXL,
    USE_FP16,
    USE_XFORMERS,
)
from utils.gpu_utils import get_seed
from PIL import Image
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    # DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    ConsistencyDecoderVAE,
)

from transformers import CLIPModel, CLIPTextConfig, CLIPTextModel, AutoTokenizer


if SD_MODEL.endswith(".safetensors") and not os.path.exists(SD_MODEL):
    raise Exception(f"Stable diffusion model not found: {SD_MODEL}")


class SDClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist

        return cls._instance

    def __init__(self):
        self.generator = get_seed(42)
        self.image_pipeline = None
        self.video_pipeline = None
        self.inpaint = None
        self.vae = None

        # Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
        self.text_encoder = CLIPTextModel(CLIPTextConfig())

        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=os.path.join("models", "CLIP")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=os.path.join("models", "CLIP")
        )
        self.controlnet_model = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16,
            # variant="fp16" if USE_FP16 else None, # No fp16 variant available for canny
            cache_dir=os.path.join("models", "ControlNet"),
        )
        self.video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if USE_FP16 else torch.float32,
            variant="fp16" if USE_FP16 else None,
            cache_dir="models/img2vid",
        )
        self.video_pipeline.to(memory_format=torch.channels_last, dtype=torch.float16)

        if torch.cuda.is_available():
            self.video_pipeline.enable_sequential_cpu_offload(0)

        image_pipeline_type = (
            StableDiffusionXLPipeline if SD_USE_SDXL else StableDiffusionPipeline
        )
        image_scheduler_type = (
            LMSDiscreteScheduler if SD_USE_SDXL else EulerDiscreteScheduler
        )

        single_file = SD_MODEL.endswith(".safetensors")

        from_model = (
            image_pipeline_type.from_single_file
            if single_file
            else image_pipeline_type.from_pretrained
        )

        if SD_USE_HYPERTILE:
            self.vae = ConsistencyDecoderVAE.from_pretrained(
                "openai/consistency-decoder",
                variant="fp16" if USE_FP16 else None,
                torch_dtype=torch.float16 if USE_FP16 else torch.float32,
                device=DEVICE,
                safetensors=True,
            )
            self.image_pipeline = from_model(
                SD_MODEL,
                variant="fp16" if USE_FP16 else None,
                safetensors=not single_file,
                enable_cuda_graph=torch.cuda.is_available(),
                # vae is predefined
            )
            self.image_pipeline.vae.disable_tiling()
        else:
            self.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                # variant="fp16" if USE_FP16 else None, # no fp16 available
                torch_dtype=torch.float16,
                safetensors=True,
                device=DEVICE,
            )
            self.image_pipeline = from_model(
                SD_MODEL,
                variant="fp16" if USE_FP16 else None,
                safetensors=not single_file,
                enable_cuda_graph=torch.cuda.is_available(),
                vae=self.vae,
            )

        self.image_pipeline.to(
            memory_format=torch.channels_last,
            dtype=torch.float16 if USE_FP16 else torch.float32,
            device=DEVICE,
        )

        if torch.cuda.is_available():
            self.image_pipeline.enable_model_cpu_offload(0)

        self.image_pipeline.scheduler = image_scheduler_type.from_config(
            self.image_pipeline.scheduler.config
        )

        self.txt2img = AutoPipelineForText2Image.from_pipe(
            self.image_pipeline, safety_checker=None, requires_safety_checker=False
        )

        self.img2img = AutoPipelineForImage2Image.from_pipe(
            self.image_pipeline, safety_checker=None, requires_safety_checker=False
        )

        self.inpaint = AutoPipelineForInpainting.from_pipe(
            self.image_pipeline, safety_checker=None, requires_safety_checker=False
        )

        self.controlnet = StableDiffusionControlNetImg2ImgPipeline(
            vae=self.image_pipeline.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.image_pipeline.unet,
            controlnet=self.controlnet_model,
            scheduler=self.image_pipeline.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

        if torch.cuda.is_available():
            self.controlnet.enable_model_cpu_offload(0)

        if USE_XFORMERS:
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

            if not SD_USE_HYPERTILE:
                self.image_pipeline.enable_xformers_memory_efficient_attention(
                    attention_op=MemoryEfficientAttentionFlashAttentionOp
                )
                self.image_pipeline.vae.enable_xformers_memory_efficient_attention(
                    attention_op=None  # skip attention op for VAE
                )
                self.video_pipeline.enable_xformers_memory_efficient_attention(
                    attention_op=None  # skip attention op for video
                )

        else:
            if not SD_USE_HYPERTILE:
                self.image_pipeline.enable_attention_slicing()
                self.video_pipeline.enable_attention_slicing()

    def upscale(
        self,
        image,
        original_width: int,
        original_height: int,
        prompt: str,
        negative_prompt: str,
        steps: int,
        use_canny: bool = False,
    ):
        upscaled_image = image.resize(
            (int(original_width * 3), int(original_height * 3)),
            Image.Resampling.NEAREST,
        )

        if use_canny:
            np_img = np.array(image)
            outline = Canny(np_img, 100, 200)
            outline = outline[:, :, None]
            outline = np.concatenate([outline, outline, outline], axis=2)
            canny_image = Image.fromarray(outline)
            canny_image.save("canny.png")

            return self.controlnet(
                prompt=prompt,
                image=outline,
                control_image=image,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                strength=1,
                generator=self.generator,
            ).images[0]
        else:
            return self.img2img(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=upscaled_image,
                num_inference_steps=steps,
                strength=1,
                generator=self.generator,
            ).images[0]
