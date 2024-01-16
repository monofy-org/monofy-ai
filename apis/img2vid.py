import logging
import time
import imageio
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
import numpy as np
from PIL import Image
from settings import SD_USE_HYPERTILE_VIDEO
from hyper_tile import split_attention
from urllib.parse import unquote
from utils.file_utils import random_filename
from utils.image_utils import load_image
from utils.gpu_utils import load_gpu_task, set_seed, gpu_thread_lock
from utils.misc_utils import print_completion_time
from utils.video_utils import frames_to_video

router = APIRouter()

MAX_FRAMES = 25


@router.get("/img2vid")
async def img2vid(
    image_url: str,
    motion_bucket: int = 127,
    steps: int = 10,
    width: int = 512,
    height: int = 512,
    fps: int = 6,
    frames: int = MAX_FRAMES,
    noise: float = 0,
    interpolate: int = 2,  # experimental
    seed: int = -1,
    audio_url: str = None,
):
    async with gpu_thread_lock:
        start_time = time.time()
        from clients import SDClient

        load_gpu_task("img2vid", SDClient)

        url = unquote(image_url)
        image = load_image(url)

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
            filename_noext = random_filename(None, True)
            filename = f"{filename_noext}-0.mp4"

            import modules.rife

            if interpolate > 0:
                frames = modules.rife.interpolate(
                    frames, count=interpolate, scale=1, pad=1, change=0
                )

            with imageio.get_writer(
                filename, format="mp4", fps=fps * interpolate
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
            set_seed(seed)
            video_frames = SDClient.img2vid_pipeline(
                image,
                decode_chunk_size=25,
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
                SDClient.img2vid_pipeline.vae,
                tile_size=256,
                aspect_ratio=aspect_ratio,
            )
            split_unet = split_attention(
                SDClient.img2vid_pipeline.unet,
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