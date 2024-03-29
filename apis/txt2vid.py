import asyncio
import gc
import logging
import time
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
import imageio
import numpy as np
import torch
from utils.gpu_utils import load_gpu_task, gpu_thread_lock
from utils.misc_utils import print_completion_time
from utils.video_utils import frames_to_video
from utils.file_utils import random_filename

router = APIRouter()

MAX_FRAMES = 25

@router.get("/txt2vid")
async def txt2vid(
    prompt: str,
    width: int = 512,
    height: int = 512,
    steps: int = 10,
    audio_url: str = None,
    frames: int = MAX_FRAMES,
    fps: int = 4,
    interpolate: int = 3,
):    
    await asyncio.sleep(0.1)

    async with gpu_thread_lock:
        start_time = time.time()
        from clients import SDClient
        # import modules.rife

        load_gpu_task("txt2vid", SDClient)

        if not SDClient.pipelines["txt2vid"]:
            SDClient.init_txt2vid()

        filename_noext = random_filename()
        filename = f"{filename_noext}-0.mp4"

        num_frames = min(frames, MAX_FRAMES)

        frames = SDClient.pipelines["txt2vid"](
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            num_frames=num_frames,
        ).frames

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()            

        if interpolate > 0:
            #frames = modules.rife.interpolate(
            #    frames, count=interpolate, scale=1, pad=1, change=0.3
            #)
            from submodules.frame_interpolation.eval.util import interpolate_recursively_from_memory
            frames = interpolate_recursively_from_memory(frames, interpolate, SDClient.film_interpolator)

        with imageio.get_writer(
            filename, format="mp4", fps=fps * interpolate
        ) as video_writer:
            for frame in frames:
                try:
                    video_writer.append_data(np.array(frame))
                except Exception as e:
                    logging.error(e)

        if audio_url:
            frames_to_video(
                filename, filename, fps=fps * interpolate, audio_url=audio_url
            )

        print_completion_time(start_time)

        return FileResponse(
            filename, filename=f"{filename_noext}.mp4", media_type="video/mp4"
        )
