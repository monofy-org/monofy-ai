import logging
import os

import imageio
import numpy as np
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
from PIL import Image

from classes.requests import Txt2VidRequest
from modules.plugins import PluginBase, release_plugin, use_plugin_unsafe
from plugins.mmaudio import MMAudioPlugin, MMAudioRequest
from settings import CACHE_PATH
from utils.audio_utils import get_audio_from_request
from utils.file_utils import delete_file, random_filename
from utils.video_utils import replace_audio

# from submodules.frame_interpolation.eval.interpolator import Interpolator
# from submodules.frame_interpolation.eval.util import (
#     interpolate_recursively_from_memory,
# )


class VideoPlugin(PluginBase):
    name = "Video Interpolation"
    description = "Interpolate frames in a video"

    def __init__(self):
        super().__init__()
        self.has_film = os.path.exists("models/film_net/Style/saved_model")
        # if self.has_film:

        #     self.resources["film_interpolator"] = (
        #         Interpolator("models/film_net/Style/saved_model", None)
        #         if self.has_film
        #         else None
        #     )

    def unload(self):
        import modules.rife

        modules.rife.unload()

    def video_response(
        self,
        background_tasks: BackgroundTasks,
        frames: list[Image.Image],
        req: Txt2VidRequest,
        return_path=False,
        previous_frames: list[Image.Image] = [],
    ):
        if req.interpolate_film > 0 or req.interpolate_rife > 0:
            frames = self.interpolate_frames(
                frames, req.interpolate_film, req.interpolate_rife
            )
        if req.fast_interpolate:
            new_frames = []
            for i in range(0, len(frames) - 1):
                if not isinstance(frames[i], Image.Image):
                    frames[i] = Image.fromarray(frames[i])
                if not isinstance(frames[i + 1], Image.Image):
                    frames[i + 1] = Image.fromarray(frames[i + 1])
                new_frames.append(frames[i])
                new_frame = Image.blend(frames[i], frames[i + 1], 0.3)
                new_frames.append(new_frame)

            new_frames.append(frames[-1])
            frames = new_frames

        filename = random_filename("mp4", False)
        full_path = os.path.join(CACHE_PATH, filename)

        fps = req.fps if req.fps else 6
        if req.interpolate_rife > 0:
            fps = fps * (req.interpolate_rife + 1)

        writer = imageio.get_writer(full_path, format="mp4", fps=fps)

        if previous_frames:
            connecting_frames = [previous_frames[-2], frames[0]]
            stitch_frames = self.interpolate_frames(connecting_frames, 0, 2)
            frames = previous_frames[0:-2] + stitch_frames + frames[1:]

        for frame in frames:
            writer.append_data(np.array(frame))

        writer.close()

        
        from ffmpy import FFmpeg

        # Create new output path
        output_path = random_filename("mp4", False)
        output_path = os.path.join(CACHE_PATH, output_path)

        # Apply minterpolate filter using ffmpy
        ff = FFmpeg(            
            inputs={full_path: None},
            outputs={output_path: '-vf minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1'}
        )
        ff.run()
        # Remove original file and update full_path
        if os.path.exists(full_path):
            os.remove(full_path)
        full_path = output_path
        

        mmaudio: str | None = None
        if req.mmaudio_prompt or req.mmaudio_negative_prompt:
            try:
                plugin: MMAudioPlugin = use_plugin_unsafe(MMAudioPlugin)
                mmaudio, _ = plugin.generate(
                    MMAudioRequest(
                        prompt=req.mmaudio_prompt,
                        negative_prompt=req.mmaudio_negative_prompt,
                        video=full_path,
                        audio_only=True,
                    )
                )
                new_path = random_filename("mp4")
                replace_audio(full_path, mmaudio, new_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
                if os.path.exists(mmaudio):
                    os.remove(mmaudio)
                full_path = new_path
            except Exception as e:
                logging.error(e)

        elif req.audio:
            audio_path = get_audio_from_request(req.audio)
            new_path = random_filename("mp4")
            replace_audio(full_path, audio_path, new_path)
            if os.path.exists(full_path):
                os.remove(full_path)
            if os.path.exists(audio_path):
                os.remove(audio_path)
            full_path = new_path

        # if background_tasks:
        #     if audio and audio_path and os.path.exists(full_path):
        #         background_tasks.add_task(delete_file, audio_path)
        #     if full_path and os.path.exists(full_path):
        #         background_tasks.add_task(delete_file, full_path)

        if return_path:
            return full_path

        return FileResponse(
            full_path,
            media_type="video/mp4",
            filename=filename,
            headers={"Content-Length": str(os.path.getsize(full_path))},
        )

    def interpolate_frames(
        self, frames: list, interpolate_film: int = 1, interpolate_rife: int = 0
    ):
        logging.info(f"Interpolating {len(frames)} frames x{interpolate_film}...")

        # if interpolate_film > 0:
        #     if self.resources.get("film_interpolator") is not None:
        #         logging.info("Using FiLM model for video frame interpolation.")
        #         film_interpolator = self.resources["film_interpolator"]

        #         import tensorflow as tf
        #         frames = [
        #             tf.image.convert_image_dtype(np.array(frame), tf.float32)
        #             for frame in frames
        #         ]

        #         frames = list(
        #             interpolate_recursively_from_memory(
        #                 frames, interpolate_film, film_interpolator
        #             )
        #         )

        #         frames = np.clip(frames, 0, 1)

        #         # Convert from tf.float32 to np.uint8
        #         frames = [
        #             Image.fromarray(np.array(frame * 255).astype(np.uint8))
        #             for frame in frames
        #         ]

        #         tf.keras.backend.clear_session()

        #     else:
        #         logging.error("FiLM model not found. Skipping FiLM interpolation.")
        #         interpolate_film = 0

        if interpolate_rife > 0:
            logging.info("Using RIFE model for video frame interpolation.")
            import modules.rife

            frames = modules.rife.interpolate(
                frames, count=interpolate_rife, scale=1, pad=1
            )

        return frames
