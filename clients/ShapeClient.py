import logging
import os
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif, export_to_ply
import numpy as np
import torch
import trimesh
from settings import USE_XFORMERS
from utils.gpu_utils import (
    load_gpu_task,
    autodetect_device,
    autodetect_dtype,
    is_fp16_available,
)
from clients import ShapeClient

friendly_name = "shap-e"
logging.warn(f"Initializing {friendly_name}...")


def export_to_glb(ply_path, file_path):
    mesh = trimesh.load(ply_path)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot)
    mesh_export = mesh.export(file_path, file_type="glb")
    return mesh_export


pipe = ShapEPipeline.from_pretrained(
    "openai/shap-e",
    device=autodetect_device(),
    variant="fp16" if is_fp16_available else None,
    cache_dir=os.path.join("models", "Shap-E"),
)
pipe.to(dtype=autodetect_dtype(), memory_format=torch.channels_last)

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
    load_gpu_task(friendly_name, ShapeClient)
    if format == "gif":
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            frame_size=256,
        ).images[0]
    elif format == "ply" or format == "glb":
        images = pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            frame_size=256,
            output_type="mesh",
        ).images[0]
    else:
        return None

    if format == "ply" or format == "glb":
        ply_path = f"{file_path}.ply"
        export_to_ply(images, ply_path)

        if format == "glb":
            file_path = f"{file_path}.glb"
            export_to_glb(ply_path, file_path)
        else:
            file_path = ply_path

    else:
        file_path = f"{file_path}.gif"
        export_to_gif(images, file_path)

    print(f"Saving to {file_path}...")

    return file_path


def offload(for_task: str):
    global friendly_name
    pipe.maybe_free_model_hooks()
    pass
