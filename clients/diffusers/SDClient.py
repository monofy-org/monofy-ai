import os
import torch
from diffusers import StableVideoDiffusionPipeline, AutoencoderTiny
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

        self.video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if USE_FP16 else torch.float32,
            variant="fp16" if USE_FP16 else None,
            cache_dir="models/img2vid",
            safetensors=True,
            device=DEVICE,
        )
        self.video_pipeline.to(memory_format=torch.channels_last, dtype=torch.float16)
        self.video_pipeline.enable_sequential_cpu_offload(0)

        self.video_pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.video_pipeline.scheduler.config
        )

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
                safetensors=True
            )
            self.image_pipeline = from_model(
                SD_MODEL,
                variant="fp16" if USE_FP16 else None,
                safetensors=not single_file,
                enable_cuda_graph=torch.cuda.is_available(),
                #vae=self.vae # hypertile handles VAE
            )
        else:
            self.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                #variant="fp16" if USE_FP16 else None, # no fp16 available
                torch_dtype=torch.float16,
                safetensors=True,
                device=DEVICE,                
            )
            
            self.image_pipeline = from_model(
                SD_MODEL,
                variant="fp16" if USE_FP16 else None,
                safetensors=not single_file,
                enable_cuda_graph=torch.cuda.is_available(),
                vae=self.vae
            )


        self.image_pipeline.to(
            memory_format=torch.channels_last,
            dtype=torch.float16 if USE_FP16 else torch.float32,
            device=DEVICE,
        )

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
    ):
        upscaled_image = image.resize(
            (int(original_width * 1.25 * 2), int(original_height * 1.25 * 2)),
            Image.Resampling.NEAREST,
        )
        return self.img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upscaled_image,
            num_inference_steps=steps,
            strength=1,
            generator=self.generator,
        ).images[0]
