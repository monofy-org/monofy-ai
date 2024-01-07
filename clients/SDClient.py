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
    USE_DEEPSPEED,
    USE_XFORMERS,
)
from utils.file_utils import fetch_pretrained_model
from utils.gpu_utils import (
    autodetect_device,
    autodetect_dtype,
    get_seed,
    is_fp16_available,
)
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    LMSDiscreteScheduler,
    # ConsistencyDecoderVAE,
    StableVideoDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
    # AutoencoderKLTemporalDecoder,
    # AutoencoderTiny,
)

from transformers import CLIPTextConfig, CLIPTextModel, CLIPProcessor, AutoTokenizer

from utils.image_utils import create_upscale_mask

is_fp16_available

if SD_MODEL.endswith(".safetensors") and not os.path.exists(SD_MODEL):
    raise Exception(f"Stable diffusion model not found: {SD_MODEL}")

img2vid_model_path = fetch_pretrained_model(
    "stabilityai/stable-video-diffusion-img2vid-xt", "img2vid"
)

friendly_name = "stable diffusion"
logging.warn(f"Initializing {friendly_name}...")
device = autodetect_device()
dtype = autodetect_dtype()
image_pipeline = None
img2vid_pipeline: StableVideoDiffusionPipeline = None
txt2vid_pipeline: DiffusionPipeline = None

inpaint = None
vae = AutoencoderKL(force_upcast=NO_HALF_VAE)
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

model_name = (
    "openai/clip-vit-base-patch16"
    if is_fp16_available
    else "openai/clip-vit-base-patch32"
)
clip_model = fetch_pretrained_model(model_name, "CLIP")

#processor = CLIPProcessor.from_pretrained(
#    clip_model, cache_dir=os.path.join("models", "CLIP")
#)

tokenizer = AutoTokenizer.from_pretrained(clip_model)

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
    cache_dir="models/img2vid",
)
#img2vid_pipeline.to(device, memory_format=torch.channels_last)
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
image_scheduler_type = LMSDiscreteScheduler if SD_USE_SDXL else DPMSolverSDEScheduler

single_file = SD_MODEL.endswith(".safetensors")

from_model = (
    image_pipeline_type.from_single_file
    if single_file
    else image_pipeline_type.from_pretrained
)

if SD_USE_HYPERTILE:
    image_pipeline = from_model(
        SD_MODEL,        
        variant="fp16" if is_fp16_available else None,
        torch_dtype=dtype,
        safetensors=not single_file,
        enable_cuda_graph=torch.cuda.is_available(),
        #vae=vae,
    )
else:
    image_pipeline = from_model(
        SD_MODEL,        
        variant="fp16" if is_fp16_available else None,
        torch_dtype=dtype,
        safetensors=not single_file,
        enable_cuda_graph=torch.cuda.is_available(),
        #vae=vae,
    )

image_pipeline.vae.force_upscale = True

if torch.cuda.is_available():
    image_pipeline.enable_model_cpu_offload()

image_pipeline.scheduler.config['lower_order_final'] = True
image_pipeline.scheduler = image_scheduler_type.from_config(
    image_pipeline.scheduler.config
)

txt2img = AutoPipelineForText2Image.from_pipe(
    image_pipeline,
    safety_checker=None,
    requires_safety_checker=False,
    device=device,
    dtype=dtype,
    vae=image_pipeline.vae,
)
txt2img.scheduler.config['lower_order_final'] = True
txt2img.scheduler = image_scheduler_type.from_config(txt2img.scheduler.config)

img2img = AutoPipelineForImage2Image.from_pipe(
    image_pipeline,
    safety_checker=None,
    requires_safety_checker=False,
    device=device,
    dtype=dtype,
    vae=image_pipeline.vae,
)
img2img.scheduler.config['lower_order_final'] = True
img2img.scheduler = image_scheduler_type.from_config(img2img.scheduler.config)

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
    tokenizer=tokenizer,
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
    return inpaint(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=get_seed(seed),
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
    upscale_coef=0,
    seed=-1,
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

        return controlnet(
            prompt=prompt,
            image=outline,
            control_image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=strength * 10,
            generator=get_seed(seed),
        ).images[0]
    else:
        upscaled_image = img2img(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upscaled_image,
            num_inference_steps=steps,
            strength=strength,
            generator=get_seed(seed),
            #width=original_width * 3,
            #height=original_height * 3,
        ).images[0]

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
    if for_task == "svd":        
        image_pipeline.maybe_free_model_hooks()
        txt2vid_pipeline.maybe_free_model_hooks()
    elif for_task == "stable diffusion":        
        img2vid_pipeline.maybe_free_model_hooks()
        txt2vid_pipeline.maybe_free_model_hooks()
