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
    USE_DEEPSPEED,
    USE_XFORMERS,
)
from utils.file_utils import fetch_pretrained_model
from utils.gpu_utils import (
    autodetect_device,
    autodetect_dtype,
    set_seed,
    is_fp16_available,
)
from PIL import Image
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers import (
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
    # AutoencoderKLTemporalDecoder,
    # AutoencoderTiny,
)
from transformers import CLIPTextConfig, CLIPTextModel
from utils.image_utils import create_upscale_mask

from insightface.app import FaceAnalysis
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID


friendly_name = "stable diffusion"
logging.warn(f"Initializing {friendly_name}...")
device = autodetect_device()
dtype = autodetect_dtype()
image_pipeline = None
img2vid_pipeline: StableVideoDiffusionPipeline = None
txt2vid_pipeline: DiffusionPipeline = None
inpaint = None


if SD_MODEL.endswith(".safetensors") and not os.path.exists(SD_MODEL):
    raise Exception(f"Stable diffusion model not found: {SD_MODEL}")

img2vid_model_path = fetch_pretrained_model(
    "stabilityai/stable-video-diffusion-img2vid-xt", "img2vid"
)
vae_model_path = fetch_pretrained_model("stabilityai/sd-vae-ft-mse", "VAE")
image_encoder_path = fetch_pretrained_model(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "CLIP"
)

if SD_USE_VAE:
    vae = AutoencoderKL.from_pretrained(
        vae_model_path, cache_dir=os.path.join("models", "VAE")
    )
    vae.to(
        dtype=torch.float16 if is_fp16_available and not NO_HALF_VAE else torch.float32
    )


# preview_vae = preview_vae = AutoencoderTiny.from_pretrained(
#    "madebyollin/taesd",
#    # variant="fp16" if USE_FP16 else None, # no fp16 available
#    torch_dtype=dtype,
#    safetensors=True,
#    device=device,
#    cache_dir=os.path.join("models", "VAE"),
# )

# Initializing a CLIPTextModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
text_encoder = CLIPTextModel(CLIPTextConfig())

controlnet_model = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    device=device,
    torch_dtype=dtype,
    # variant="fp16" if USE_FP16 else None, # No fp16 variant available for canny
    cache_dir=os.path.join("models", "ControlNet"),
)

video_dtype = (
    torch.float16 if is_fp16_available else torch.float32
)  # bfloat16 not available
img2vid_pipeline = StableVideoDiffusionPipeline.from_pretrained(
    img2vid_model_path,
    torch_dtype=video_dtype,
    variant="fp16" if is_fp16_available else None,
    cache_dir=os.path.join("models", "img2vid"),
)
# img2vid_pipeline.to(device, memory_format=torch.channels_last)
# img2vid_pipeline.vae.force_upscale = True
# img2vid_pipeline.vae.to(device=device, dtype=video_dtype)

txt2vid_pipeline = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    cache_dir=os.path.join("models", "txt2vid"),
    torch_dtype=dtype,
)

if torch.cuda.is_available():
    img2vid_pipeline.enable_sequential_cpu_offload()
    txt2vid_pipeline.enable_model_cpu_offload()

image_pipeline_type = (
    StableDiffusionXLPipeline if SD_USE_SDXL else StableDiffusionPipeline
)

single_file = SD_MODEL.endswith(".safetensors")

from_model = (
    image_pipeline_type.from_single_file
    if single_file
    else image_pipeline_type.from_pretrained
)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

if SD_USE_HYPERTILE:
    image_pipeline = from_model(
        SD_MODEL,
        # variant="fp16" if not single_file and is_fp16_available else None,
        torch_dtype=torch.float16 if is_fp16_available else torch.float32,
        safetensors=not single_file,
        enable_cuda_graph=torch.cuda.is_available(),
        vae=vae if SD_USE_VAE else None,
        feature_extractor=None,
    )
else:
    image_pipeline = from_model(
        SD_MODEL,
        # variant="fp16" if not single_file and is_fp16_available else None,
        torch_dtype=torch.float16 if is_fp16_available else torch.float32,
        safetensors=not single_file,
        enable_cuda_graph=torch.cuda.is_available(),
        vae=vae if SD_USE_VAE else None,
        feature_extractor=None,
    )


# face_app_path = fetch_pretrained_model("h94/IP-Adapter-FaceID", "IP-Adapter-FaceID")
# face_app = FaceAnalysis(
#    name="buffalo_s", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
# )
# face_app.prepare(ctx_id=0, det_size=(640, 640))
# ip_ckpt = os.path.join(
#    face_app_path,
#    "ip-adapter-faceid_sd15.bin"
# )
# ip_model = IPAdapterFaceID(image_pipeline, ip_ckpt, device)

image_pipeline.vae.force_upscale = True
image_pipeline.vae.use_tiling = False

if torch.cuda.is_available():
    image_pipeline.enable_model_cpu_offload()

image_pipeline.scheduler.config["lower_order_final"] = True
image_pipeline.scheduler.config["use_karras_sigmas"] = True

schedulers = {}
schedulers["euler"] = EulerDiscreteScheduler.from_config(
    image_pipeline.scheduler.config
)
schedulers["euler_a"] = EulerAncestralDiscreteScheduler.from_config(
    image_pipeline.scheduler.config
)
schedulers["sde"] = DPMSolverSDEScheduler.from_config(image_pipeline.scheduler.config)
schedulers["lms"] = LMSDiscreteScheduler.from_config(image_pipeline.scheduler.config)
schedulers["heun"] = HeunDiscreteScheduler.from_config(image_pipeline.scheduler.config)
schedulers["ddim"] = DDIMScheduler.from_config(image_pipeline.scheduler.config)

txt2img = AutoPipelineForText2Image.from_pipe(
    image_pipeline,
    safety_checker=None,
    requires_safety_checker=False,
    device=device,
    dtype=dtype,
    # vae=vae,
)
txt2img.scheduler.config["lower_order_final"] = True
txt2img.scheduler = schedulers[SD_DEFAULT_SCHEDULER]

img2img = AutoPipelineForImage2Image.from_pipe(
    image_pipeline,
    safety_checker=None,
    requires_safety_checker=False,
    device=device,
    dtype=dtype,
    # vae=vae,
)
img2img.scheduler.config["lower_order_final"] = True
img2img.scheduler = schedulers[SD_DEFAULT_SCHEDULER]

inpaint = AutoPipelineForInpainting.from_pipe(
    image_pipeline,
    safety_checker=None,
    requires_safety_checker=False,
    device=device,
    dtype=dtype,
)

controlnet = StableDiffusionControlNetImg2ImgPipeline(
    vae=image_pipeline.vae,
    text_encoder=text_encoder,
    tokenizer=image_pipeline.tokenizer,
    unet=image_pipeline.unet,
    controlnet=controlnet_model,
    scheduler=image_pipeline.scheduler,
    safety_checker=None,
    # image_processor=image_pipeline.image_processor,
    feature_extractor=image_pipeline.image_processor,
    requires_safety_checker=False,
)

if torch.cuda.is_available():
    controlnet.enable_model_cpu_offload()

if USE_XFORMERS:
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

    if not SD_USE_HYPERTILE and not USE_DEEPSPEED:
        image_pipeline.enable_xformers_memory_efficient_attention(
            attention_op=MemoryEfficientAttentionFlashAttentionOp
        )
        image_pipeline.vae.enable_xformers_memory_efficient_attention(
            attention_op=None  # skip attention op for VAE
        )
    if not USE_DEEPSPEED:
        img2vid_pipeline.enable_xformers_memory_efficient_attention(
            attention_op=None  # skip attention op for video
        )
        txt2vid_pipeline.enable_xformers_memory_efficient_attention(
            attention_op=None  # skip attention op for video
        )

else:
    if not SD_USE_HYPERTILE:
        image_pipeline.enable_attention_slicing()
        img2vid_pipeline.enable_attention_slicing()
        txt2vid_pipeline.enable_attention_slicing()


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
    global inpaint
    mask_image = create_upscale_mask(width, height, aspect_ratio)
    set_seed(seed)
    return inpaint(
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
    use_canny: bool = False,
    upscale_coef: float = 0,
    seed: int = -1,
):
    global img2img
    global controlnet

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

    if use_canny:
        np_img = np.array(image)
        outline = Canny(np_img, 100, 200)
        outline = outline[:, :, None]
        outline = np.concatenate([outline, outline, outline], axis=2)
        canny_image = Image.fromarray(outline)
        canny_image.save("canny.png")

        set_seed(seed)

        return controlnet(
            prompt=prompt,
            image=outline,
            control_image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=strength * 10,
        ).images[0]
    else:
        set_seed(seed)
        upscaled_image = img2img(
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
    global img2vid_pipeline
    logging.info("Offloading diffusers...")
    if for_task == "txt2vid":
        image_pipeline.maybe_free_model_hooks()
        img2vid_pipeline.maybe_free_model_hooks()
    if for_task == "img2vid":
        image_pipeline.maybe_free_model_hooks()
        txt2vid_pipeline.maybe_free_model_hooks()
    elif for_task == "stable diffusion":
        img2vid_pipeline.maybe_free_model_hooks()
        txt2vid_pipeline.maybe_free_model_hooks()
