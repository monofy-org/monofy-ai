import gc
import logging
import os
from diffusers.models import ControlNetModel
import cv2
import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
from utils.gpu_utils import (
    autodetect_device,
    autodetect_dtype,
    load_gpu_task,
    set_seed,
    gpu_thread_lock,
)
from insightface.app import FaceAnalysis
from settings import (
    SD_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_HEIGHT,
    SD_DEFAULT_MODEL_INDEX,    
    SD_DEFAULT_SCHEDULER,
    SD_DEFAULT_STEPS,    
    SD_DEFAULT_WIDTH,
    SD_MODELS,
    USE_XFORMERS,
)
from submodules.InstantID.pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)
from clients import IPAdapterClient


FACEID_MODEL = "InstantX/InstantID"
model_root = os.path.join("models", FACEID_MODEL)
app: FaceAnalysis = None
pipe: StableDiffusionXLInstantIDPipeline = None
controlnet: ControlNetModel = None


hf_hub_download(
    repo_id=FACEID_MODEL,
    filename="ControlNetModel/config.json",
    local_dir=model_root,
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id=FACEID_MODEL,
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir=model_root,
    local_dir_use_symlinks=False,
)
hf_hub_download(
    repo_id=FACEID_MODEL,
    filename="ip-adapter.bin",
    local_dir=model_root,
    local_dir_use_symlinks=False,
)


def load_model():
    logging.warning("Loading ip-adapter")

    global app
    global pipe
    global controlnet

    device = autodetect_device()
    dtype = autodetect_dtype(True)

    app = FaceAnalysis(
        name="buffalo_s",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    controlnet_path = os.path.join(model_root, "ControlNetModel")
    # load IdentityNet
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, device=device, torch_dtype=dtype
    )
    controlnet.eval()

    kwargs = dict(controlnet=controlnet, dtype=dtype, device=device)

    model = SD_MODELS[SD_DEFAULT_MODEL_INDEX]

    single_file = model.endswith(".safetensors")

    if single_file:
        pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
            SD_MODELS[SD_DEFAULT_MODEL_INDEX], **kwargs
        )
    else:        
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            model,
            local_dir=os.path.join("models", "Stable-diffusion", model),
            local_dir_use_symlinks=False,
        )

    if torch.cuda.is_available():
        pipe.cuda()
        if USE_XFORMERS:
            pipe.set_use_memory_efficient_attention_xformers(True)

    face_adapter_path = os.path.join(model_root, "ip-adapter.bin")
    pipe.load_ip_adapter_instantid(face_adapter_path)


def unload():
    logging.info("Unloading ip-adapter")
    global app
    global pipe
    global controlnet

    app = None
    pipe = None
    controlnet = None
    gc.collect()
    torch.cuda.empty_cache()


async def generate(
    face_image: Image,
    prompt: str = "",
    negative_prompt: str = "",
    num_inference_steps: int = SD_DEFAULT_STEPS,
    guidance_scale: float = SD_DEFAULT_GUIDANCE_SCALE,
    controlnet_scale: float = 0.8,
    width: int = SD_DEFAULT_WIDTH,
    height: int = SD_DEFAULT_HEIGHT,
    nsfw: bool = False,
    upscale: float = 0,
    upscale_strength: float = 0.65,
    seed: int = -1,
    scheduler: str = SD_DEFAULT_SCHEDULER,
):
    global pipe
    global app

    async with gpu_thread_lock:
        load_gpu_task("ip-adapter", IPAdapterClient)
        # Convert the prompt to lowercase for consistency

        if pipe is None:
            load_model()

        # prepare face emb
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
        )[
            -1
        ]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(face_image, face_info["kps"])

        seed = set_seed(seed)

        # generate image
        pipe.set_ip_adapter_scale(controlnet_scale)
        result: Image.Image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
        ).images[0]
        unload()
        return result


def offload(for_task: str = None):
    unload()
