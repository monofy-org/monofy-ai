import gc
import io
import logging
import os
import time
from PIL import Image
from fastapi import HTTPException, BackgroundTasks, UploadFile
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
import torch
from submodules.moondream.moondream.text_model import TextModel
from submodules.moondream.moondream.vision_encoder import VisionEncoder
from huggingface_hub import snapshot_download
from utils.gpu_utils import gpu_thread_lock, load_gpu_task
from utils.image_utils import fetch_image
from utils.misc_utils import print_completion_time
from apis import vision

router = APIRouter()


IMAGE_DETECT_MODEL = "vikhyatk/moondream1"

vision_encoder = None
text_model = None


def load_model():
    global vision_encoder
    global text_model

    model_path = os.path.join("models", IMAGE_DETECT_MODEL)

    if not os.path.isdir(model_path):
        model_path = snapshot_download(
            IMAGE_DETECT_MODEL,
            local_dir=model_path,
            local_dir_use_symlinks=False,
        )

    if vision_encoder is None:
        vision_encoder = VisionEncoder(model_path)

    if text_model is None:
        text_model = TextModel(model_path)


def unload_model():
    global vision_encoder
    global text_model

    if hasattr(vision_encoder, "maybe_free_model_hooks"):
        vision_encoder.model.maybe_free_model_hooks()

    if vision_encoder is not None:
        del vision_encoder.model
        vision_encoder = None

    if text_model is not None:
        del text_model.model
        text_model = None

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    logging.debug("Unloaded vision")


def offload(for_task: str):
    logging.debug("No offload available for vision, unloading model")
    unload_model()


@router.post("/vision")
@router.get("/vision")
async def deep_object_detection(
    background_tasks: BackgroundTasks,
    image: UploadFile = None,
    image_url: str = None,
    prompt: str = "Describe the image",
):
    global vision_encoder
    global text_model

    load_gpu_task("vision", vision)

    if vision_encoder is None or text_model is None:
        load_model()

    if image is not None:
        image_pil = Image.open(io.BytesIO(await image.read()))
    elif image_url is not None:
        image_pil = fetch_image(image_url)
    else:
        return HTTPException(status_code=400, detail="No image or image_url provided")

    image_embeds = vision_encoder(image_pil)

    async with gpu_thread_lock:
        start_time = time.time()

        with torch.no_grad():
            answer = text_model.answer_question(image_embeds, prompt)

        print_completion_time(start_time, "Vision")

        return JSONResponse({"response": answer})
