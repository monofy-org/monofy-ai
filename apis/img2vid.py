import asyncio
import logging
import os
import time
import imageio
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
import numpy as np
from PIL import Image
import tensorflow as tf
from settings import SD_USE_HYPERTILE_VIDEO
from hyper_tile import split_attention
from urllib.parse import unquote
from utils.file_utils import random_filename
from utils.image_utils import crop_and_resize, load_image
from utils.gpu_utils import load_gpu_task, set_seed, gpu_thread_lock
from utils.misc_utils import print_completion_time
from utils.video_utils import frames_to_video
from submodules.frame_interpolation.eval import interpolator
from submodules.frame_interpolation.eval.util import (
    interpolate_recursively_from_memory,
)

film_interpolator = (
    interpolator.Interpolator("models/film_net/Style/saved_model", None)
    if os.path.exists("models/film_net/Style/")
    else None
)

router = APIRouter()

IMG2VID_DEFAULT_FRAMES = 25
IMG2VID_DECODE_CHUNK_SIZE = 20
IMG2VID_DEFAULT_MOTION_BUCKET = 47


@router.get("/img2vid")
async def img2vid(
    image_url: str,
    motion_bucket: int = IMG2VID_DEFAULT_MOTION_BUCKET,
    steps: int = 10,
    width: int = 512,
    height: int = 512,
    fps: int = 6,
    frames: int = IMG2VID_DEFAULT_FRAMES,
    noise: float = 0,
    interpolate: int = 1,  # experimental
    seed: int = -1,
    audio_url: str = None,
):
    await asyncio.sleep(0.1)

    async with gpu_thread_lock:
        start_time = time.time()
        from clients import SDClient

        load_gpu_task("img2vid", SDClient)

        SDClient.init_img2vid()

        url = unquote(image_url)
        image = load_image(url)
        image = crop_and_resize(image, width * 2, height * 2)

        if seed == -1:
            seed = set_seed(-1)

        aspect_ratio = width / height

        if aspect_ratio < 1:  # portrait
            image = image.crop((0, 0, image.height * aspect_ratio, image.height))
        elif aspect_ratio > 1:  # landscape
            image = image.crop((0, 0, image.width, image.width / aspect_ratio))
        else:  # square
            dim = min(image.width, image.height)
            image = image.crop((0, 0, dim, dim))

        image = image.resize((width, height), Image.Resampling.BICUBIC)

        # if frames > MAX_FRAMES:
        #    frames = MAX_FRAMES

        def process_and_get_response(frames, interpolate):
            filename_noext = random_filename()
            filename = f"{filename_noext}-0.mp4"

            logging.info(f"Interpolating {len(frames)} frames...")

            if interpolate > 0:
                if film_interpolator is not None:

                    frames = [
                        tf.image.convert_image_dtype(frame, tf.float32)
                        for frame in frames
                    ]
                    frames = list(
                        interpolate_recursively_from_memory(
                            frames, interpolate, SDClient.film_interpolator
                        )
                    )
                    frames = [
                        tf.image.convert_image_dtype(frame, tf.uint8)
                        for frame in frames
                    ]
                logging.warning(
                    "Film model not found. Falling back to Rife for interpolation."
                )
                import modules.rife

                frames = modules.rife.interpolate(
                    frames, count=interpolate + 1, scale=1, pad=1, change=0
                )

            with imageio.get_writer(
                filename, format="mp4", fps=fps * (interpolate + 1)
            ) as video_writer:
                for frame in frames:
                    try:
                        video_writer.append_data(np.array(frame))
                    except Exception as e:
                        logging.error(e)

                video_writer.close()

            if audio_url:
                frames_to_video(filename, filename, fps=fps * 5, audio_url=audio_url)

            print(f"Returning {filename}...")
            return FileResponse(filename, media_type="video/mp4")

        def gen():
            video_frames = SDClient.pipelines["img2vid"](
                image,
                decode_chunk_size=IMG2VID_DECODE_CHUNK_SIZE,
                num_inference_steps=steps,
                num_frames=frames,
                width=width,
                height=height,
                motion_bucket_id=motion_bucket,
                noise_aug_strength=noise,
            ).frames[0]

            return process_and_get_response(video_frames, interpolate)

        if SD_USE_HYPERTILE_VIDEO:
            aspect_ratio = 1 if width == height else width / height
            split_vae = split_attention(
                SDClient.pipelines["img2vid"].vae,
                tile_size=256,
                aspect_ratio=aspect_ratio,
            )
            split_unet = split_attention(
                SDClient.pipelines["img2vid"].unet,
                tile_size=256,
                aspect_ratio=aspect_ratio,
            )
            with split_vae:
                with split_unet:
                    result = gen()
                    print_completion_time(start_time)
                    return result

        else:
            result = gen()
            print_completion_time(start_time)
            return result
