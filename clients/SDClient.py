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

#from insightface.app import FaceAnalysis
#from ip_adapter.ip_adapter_faceid import IPAdapterFaceID


friendly_name = "sdxl" if SD_USE_SDXL else "stable diffusion"
logging.warn(f"Initializing {friendly_name}...")
device = autodetect_device()
dtype = autodetect_dtype()

pipelines: dict[DiffusionPipeline] = {}
controlnets = {}

pipelines["img2vid"]: StableVideoDiffusionPipeline = None

pipelines["txt2vid"]: DiffusionPipeline = None

if SD_MODEL.endswith(".safetensors") and not os.path.exists(SD_MODEL):
    raise Exception(f"Stable diffusion model not found: {SD_MODEL}")

img2vid_model_path = fetch_pretrained_model(
    "stabilityai/stable-video-diffusion-img2vid-xt", "img2vid"
)
vae_model_path = fetch_pretrained_model("stabilityai/sd-vae-ft-mse", "VAE")
# image_encoder_path = fetch_pretrained_model(
#    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "CLIP"
# )

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

controlnets["canny"] = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    device=device,
    torch_dtype=dtype,
    cache_dir=os.path.join("models", "ControlNet"),
)

controlnets["depth"] = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    device=device,
    torch_dtype=dtype,
    cache_dir=os.path.join("models", "ControlNet"),
)

video_dtype = (
    torch.float16 if is_fp16_available else torch.float32
)  # bfloat16 not available
pipelines["img2vid"] = StableVideoDiffusionPipeline.from_pretrained(
    img2vid_model_path,
    torch_dtype=video_dtype,
    variant="fp16" if is_fp16_available else None,
    cache_dir=os.path.join("models", "img2vid"),
)
# pipelines["img2vid"].to(device, memory_format=torch.channels_last)
# pipelines["img2vid"].vae.force_upscale = True
# pipelines["img2vid"].vae.to(device=device, dtype=video_dtype)

pipelines["txt2vid"] = DiffusionPipeline.from_pretrained(
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

for scheduler in schedulers.values():
    scheduler.config["lower_order_final"] = True
    scheduler.config["use_karras_sigmas"] = True

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

pipelines["img2img"] = AutoPipelineForImage2Image.from_pipe(
    image_pipeline,
    safety_checker=None,
    requires_safety_checker=False,
    device=device,
    dtype=dtype,
    # vae=vae,
)
pipelines["img2img"].scheduler.config["lower_order_final"] = True
pipelines["img2img"].scheduler = schedulers[SD_DEFAULT_SCHEDULER]

pipelines["inpaint"] = AutoPipelineForInpainting.from_pipe(
    image_pipeline,
    safety_checker=None,
    requires_safety_checker=False,
    device=device,
    dtype=dtype,
)


def create_controlnet_pipeline(name: str):
    pipelines[name] = StableDiffusionControlNetImg2ImgPipeline(
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

        pipelines["canny"].enable_xformers_memory_efficient_attention()
        pipelines["depth"].enable_xformers_memory_efficient_attention()

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
