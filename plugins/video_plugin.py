import logging
import os
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
import imageio
import numpy as np
from PIL import Image
from modules.plugins import PluginBase
from utils.file_utils import delete_file, random_filename
from utils.video_utils import add_audio_to_video, get_audio_from_request


class VideoPlugin(PluginBase):
    name = "Video Interpolation"
    description = "Interpolate frames in a video"

    def __init__(self):
        super().__init__()
        self.has_film = os.path.exists("models/film_net/Style/saved_model")
        if self.has_film:
            from submodules.frame_interpolation.eval.interpolator import Interpolator

            self.resources["film_interpolator"] = (
                Interpolator("models/film_net/Style/saved_model", None)
                if self.has_film
                else None
            )

    def video_response(
        self,
        background_tasks: BackgroundTasks,
        frames: list[Image.Image],
        fps: float = 24,
        interpolate_film: int = 1,
        interpolate_rife: int = 1,
        fast_interpolate: bool = False,
        audio: str = None,
        return_path=False,
        previous_frames: list[Image.Image] = [],
    ):
        if interpolate_film > 0 or interpolate_rife > 0:
            frames = self.interpolate_frames(frames, interpolate_film, interpolate_rife)
        if fast_interpolate:
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
        full_path = os.path.join(".cache", filename)

        fps = fps * 2 if fast_interpolate else fps

        writer = imageio.get_writer(full_path, format="mp4", fps=fps)

        frames = previous_frames + frames

        for frame in frames:
            writer.append_data(np.array(frame))

        writer.close()

        if audio:
            audio_path = random_filename("wav", False)
            get_audio_from_request(audio, audio_path)
            add_audio_to_video(full_path, audio_path, full_path)

        if background_tasks:
            if audio and audio_path and os.path.exists(full_path):
                background_tasks.add_task(delete_file, audio_path)
            if full_path and os.path.exists(full_path):
                background_tasks.add_task(delete_file, full_path)

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

        if interpolate_film > 0:
            if self.resources.get("film_interpolator") is not None:
                logging.info("Using FiLM model for video frame interpolation.")
                film_interpolator = self.resources["film_interpolator"]
                import tensorflow as tf

                frames = [
                    tf.image.convert_image_dtype(np.array(frame), tf.float32)
                    for frame in frames
                ]

                from submodules.frame_interpolation.eval.util import (
                    interpolate_recursively_from_memory,
                )

                frames = list(
                    interpolate_recursively_from_memory(
                        frames, interpolate_film, film_interpolator
                    )
                )
                frames = np.clip(frames, 0, 1)

                # Convert from tf.float32 to np.uint8
                frames = [
                    Image.fromarray(np.array(frame * 255).astype(np.uint8))
                    for frame in frames
                ]

            else:
                logging.error("FiLM model not found. Skipping FiLM interpolation.")
                interpolate_film = 0

        if interpolate_rife > 0:
            logging.info("Using RIFE model for video frame interpolation.")
            import modules.rife

            frames = modules.rife.interpolate(
                frames, count=interpolate_rife + 1, scale=1, pad=1, change=0
            )

        return frames
