import os
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif, export_to_ply
import torch
from settings import DEVICE, USE_FP16, USE_XFORMERS
from utils.gpu_utils import free_vram
from clients import ShapeClient

friendly_name = "shap-e"
pipe = ShapEPipeline.from_pretrained(
    "openai/shap-e",
    device=DEVICE,
    variant="fp16" if USE_FP16 else None,
    torch_dtype=torch.float16 if USE_FP16 else torch.float32,
    cache_dir=os.path.join("models", "Shap-E"),
)
pipe.to(memory_format=torch.channels_last)

if USE_XFORMERS:
    pipe.enable_xformers_memory_efficient_attention()

if torch.cuda.is_available():
    pipe.enable_model_cpu_offload()

def generate(
    
    prompt: str,
    file_path: str,
    steps: int = 32,
    guidance_scale: float = 15.0,
    format: str = "gif",
):
    free_vram(friendly_name, ShapeClient)
    if format == "gif":
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            frame_size=256,
        ).images[0]
    elif format == "ply":
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            frame_size=256,
            output_type="mesh",
        ).images[0]
    else:
        return None

    print(f"Saving {len(images)} images to {file_path}")

    if format == "ply":
        export_to_ply(images, file_path)

    else:
        export_to_gif(images, file_path)

    return file_path

def offload( for_task):
    pipe.maybe_free_model_hooks()
    pass
