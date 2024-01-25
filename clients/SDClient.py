import gc
import logging
import numpy as np
import os
import torch
from cv2 import Canny
from settings import (
    NO_HALF_VAE,
    SD_MODEL,
    SD_USE_HYPERTILE,
    SD_USE_SDXL,
    SD_USE_VAE,
    SD_DEFAULT_SCHEDULER,
    SD_COMPILE_UNET,
    SD_COMPILE_VAE,
    USE_DEEPSPEED,
    USE_XFORMERS,
)
from utils.file_utils import fetch_pretrained_model
from utils.gpu_utils import (
    autodetect_device,
    autodetect_dtype,
    set_seed,
    use_fp16,
)
from PIL import Image
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers import (
    SchedulerMixin,
    DiffusionPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    DPMSolverSDEScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    DDIMScheduler,
    StableVideoDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from transformers import CLIPTextConfig, CLIPTextModel
from utils.image_utils import create_upscale_mask
from nudenet import NudeDetector

# from insightface.app import FaceAnalysis
# from ip_adapter.ip_adapter_faceid import IPAdapterFaceID


friendly_name = "sdxl" if SD_USE_SDXL else "stable diffusion"
logging.warn(f"Initializing {friendly_name}...")
device = autodetect_device()
dtype = autodetect_dtype()
nude_detector = NudeDetector()

pipelines: dict[DiffusionPipeline] = {}
controlnets = {}

if SD_MODEL.endswith(".safetensors") and not os.path.exists(SD_MODEL):
    raise Exception(f"Stable diffusion model not found: {SD_MODEL}")

img2vid_model_path = fetch_pretrained_model(
    "stabilityai/stable-video-diffusion-img2vid-xt", "img2vid"
)

if SD_USE_VAE:
    vae_model_path = fetch_pretrained_model("stabilityai/sd-vae-ft-mse", "VAE")
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        vae_model_path,
        cache_dir=os.path.join("models", "VAE"),
        torch_dtype=torch.float16 if use_fp16 and not NO_HALF_VAE else torch.float32,
    )

# preview_vae = AutoencoderTiny.from_pretrained(
#    "madebyollin/taesd",
#    # variant="fp16" if USE_FP16 else None, # no fp16 available
#    torch_dtype=dtype,
#    safetensors=True,
#    device=device,
#    cache_dir=os.path.join("models", "VAE"),
# )

# Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
text_encoder = CLIPTextModel(CLIPTextConfig())
text_encoder.to(device=device, dtype=dtype)

controlnet_canny: ControlNetModel = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    device=device,
    torch_dtype=dtype,
    cache_dir=os.path.join("models", "ControlNet"),
)

controlnet_depth: ControlNetModel = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    device=device,
    torch_dtype=dtype,
    cache_dir=os.path.join("models", "ControlNet"),
)

controlnets["canny"] = controlnet_canny
controlnets["depth"] = controlnet_depth

video_dtype = (
    torch.float16 if use_fp16 else torch.float32
)  # bfloat16 not available for video

pipelines[
    "img2vid"
]: StableVideoDiffusionPipeline = StableVideoDiffusionPipeline.from_pretrained(
    img2vid_model_path,
    cache_dir=os.path.join("models", "img2vid"),
    torch_dtype=video_dtype,
    variant="fp16" if use_fp16 else None,
)
# pipelines["img2vid"].to(device, memory_format=torch.channels_last)
# pipelines["img2vid"].vae.force_upscale = True
# pipelines["img2vid"].vae.to(device=device, dtype=video_dtype)

pipelines["txt2vid"]: DiffusionPipeline = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    cache_dir=os.path.join("models", "txt2vid"),
    torch_dtype=dtype,
)

image_pipeline_type = (
    StableDiffusionXLPipeline if SD_USE_SDXL else StableDiffusionPipeline
)

single_file = SD_MODEL.endswith(".safetensors")

from_model = (
    image_pipeline_type.from_single_file
    if single_file
    else image_pipeline_type.from_pretrained
)

image_pipeline = from_model(
    SD_MODEL,
    # variant="fp16" if not single_file and is_fp16_available else None,
    torch_dtype=torch.float16 if use_fp16 else torch.float32,
    # safetensors=True,  # not single_file,
    # enable_cuda_graph=torch.cuda.is_available(),
    # vae=vae if SD_USE_VAE else None,
    feature_extractor=None,
    cache_dir=os.path.join("models", "sd" if not SD_USE_SDXL else "sdxl"),
)
image_pipeline.unet.eval()
image_pipeline.vae.eval()

# compile model (linux only)
if not os.name == "nt":
    if SD_COMPILE_UNET:
        image_pipeline.unet = torch.compile(image_pipeline.unet).to(device)
    if SD_COMPILE_VAE:
        image_pipeline.vae = torch.compile(image_pipeline.vae).to(device)

image_pipeline.vae.force_upscale = True
image_pipeline.vae.use_tiling = False

if torch.cuda.is_available():
    image_pipeline.enable_model_cpu_offload()

schedulers: dict[SchedulerMixin] = {}
schedulers["euler"]: EulerDiscreteScheduler = EulerDiscreteScheduler.from_config(
    image_pipeline.scheduler.config
)
schedulers[
    "euler_a"
]: EulerAncestralDiscreteScheduler = EulerAncestralDiscreteScheduler.from_config(
    image_pipeline.scheduler.config
)
schedulers["sde"]: DPMSolverSDEScheduler = DPMSolverSDEScheduler.from_config(
    image_pipeline.scheduler.config
)
schedulers["lms"]: LMSDiscreteScheduler = LMSDiscreteScheduler.from_config(
    image_pipeline.scheduler.config
)
schedulers["heun"]: HeunDiscreteScheduler = HeunDiscreteScheduler.from_config(
    image_pipeline.scheduler.config
)
schedulers["ddim"]: DDIMScheduler = DDIMScheduler.from_config(
    image_pipeline.scheduler.config
)

for scheduler in schedulers.values():
    scheduler.config["lower_order_final"] = not SD_USE_SDXL
    scheduler.config["use_karras_sigmas"] = True

pipelines["txt2img"]: AutoPipelineForText2Image = AutoPipelineForText2Image.from_pipe(
    image_pipeline,
    device=device,
    dtype=dtype,
    scheduler=schedulers[SD_DEFAULT_SCHEDULER],
    # vae=vae,
)

pipelines["img2img"]: AutoPipelineForImage2Image = AutoPipelineForImage2Image.from_pipe(
    image_pipeline,
    device=device,
    dtype=dtype,
    scheduler=schedulers[SD_DEFAULT_SCHEDULER],
    # vae=vae,
)

pipelines["inpaint"]: AutoPipelineForInpainting = AutoPipelineForInpainting.from_pipe(
    image_pipeline,
    device=device,
    dtype=dtype,
    scheduler=schedulers[SD_DEFAULT_SCHEDULER],
)


def censor(temp_path):
    return nude_detector.censor(
        temp_path,
        [
            "ANUS_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
        ],
    )


def create_controlnet_pipeline(name: str):
    pipelines[
        name
    ]: StableDiffusionControlNetImg2ImgPipeline = StableDiffusionControlNetImg2ImgPipeline(
        vae=image_pipeline.vae,
        text_encoder=text_encoder,
        tokenizer=image_pipeline.tokenizer,
        unet=image_pipeline.unet,
        controlnet=controlnets[name],
        scheduler=image_pipeline.scheduler,
        safety_checker=None,
        # image_processor=image_pipeline.image_processor,
        feature_extractor=image_pipeline.image_processor,
        requires_safety_checker=False,
    )


create_controlnet_pipeline("canny")
create_controlnet_pipeline("depth")

if torch.cuda.is_available():
    pipelines["img2vid"].enable_sequential_cpu_offload()
    pipelines["txt2vid"].enable_model_cpu_offload()
    pipelines["canny"].enable_model_cpu_offload()

if USE_XFORMERS:
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

    if not USE_DEEPSPEED:
        image_pipeline.enable_xformers_memory_efficient_attention(
            attention_op=MemoryEfficientAttentionFlashAttentionOp
        )
        image_pipeline.vae.enable_xformers_memory_efficient_attention(
            attention_op=None  # skip attention op for VAE
        )
        pipelines["img2vid"].enable_xformers_memory_efficient_attention(
            attention_op=None  # skip attention op for video
        )
        pipelines["txt2vid"].enable_xformers_memory_efficient_attention(
            attention_op=None  # skip attention op for video
        )

        controlnet_canny.enable_xformers_memory_efficient_attention()
        controlnet_depth.enable_xformers_memory_efficient_attention()

else:
    if not SD_USE_HYPERTILE:
        image_pipeline.enable_attention_slicing()
        pipelines["img2vid"].enable_attention_slicing()
        pipelines["txt2vid"].enable_attention_slicing()


def widen(
    image,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    aspect_ratio: float,
    seed: int = -1,
):
    mask_image = create_upscale_mask(width, height, aspect_ratio)
    set_seed(seed)
    return pipelines["inpaint"](
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        mask_image=mask_image,
    )


def upscale(
    image,
    original_width: int,
    original_height: int,
    prompt: str,
    negative_prompt: str,
    steps: int,
    strength: float = 0.6,
    controlnet: str = "canny",
    upscale_coef: float = 0,
    seed: int = -1,
):
    if steps > 100:
        logging.warn(f"Limiting steps to 100 from {steps}")
        steps = 100

    if strength > 2:
        logging.warn(f"Limiting strength to 2 from {strength}")
        strength = 2
    upscaled_image = image.resize(
        (int(original_width * upscale_coef), int(original_height * upscale_coef)),
        Image.Resampling.BICUBIC,
    )

    set_seed(seed)

    if controlnet == "canny":
        np_img = np.array(image)
        outline = Canny(np_img, 100, 200)
        outline = outline[:, :, None]
        outline = np.concatenate([outline, outline, outline], axis=2)

        # DEBUG
        canny_image = Image.fromarray(outline)
        canny_image.save("canny.png")

        return pipelines["canny"](
            prompt=prompt,
            image=outline,
            control_image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=strength * 10,
        ).images[0]
    else:
        upscaled_image = pipelines["img2img"](
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upscaled_image,
            num_inference_steps=steps,
            strength=strength,
        ).images[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return upscaled_image


def offload(for_task: str):
    global friendly_name
    global image_pipeline

    logging.info("Offloading diffusers...")
    if for_task == "txt2vid":
        image_pipeline.maybe_free_model_hooks()
        pipelines["img2vid"].maybe_free_model_hooks()
    if for_task == "img2vid":
        image_pipeline.maybe_free_model_hooks()
        pipelines["txt2vid"].maybe_free_model_hooks()
    elif for_task == friendly_name:
        pipelines["img2vid"].maybe_free_model_hooks()
        pipelines["txt2vid"].maybe_free_model_hooks()
