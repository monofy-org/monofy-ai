import logging
import os
from flask import request
import imageio
import numpy as np
import requests
from PIL import Image
from utils.file_utils import delete_file, random_filename
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse


def extract_frames(
    video_path, num_frames, trim_start=0, trim_end=0, return_json: bool = False
):

    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(video_path)
    duration = clip.duration

    start_time = trim_start
    end_time = duration - trim_end

    frame_duration = (end_time - start_time) / num_frames

    frames = []

    for i in range(num_frames):
        t = (i * frame_duration) + start_time
        image = Image.fromarray(clip.get_frame(t))
        if return_json:
            frames.append({"time": t, "image": image, "summary": None})
        else:
            frames.append(image)

    clip.close()

    return frames


def add_audio_to_video(video_path, audio_path, output_path):

    from moviepy.editor import VideoFileClip, AudioFileClip

    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    video_clip: VideoFileClip = video_clip.set_audio(audio_clip)

    video_clip.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=video_clip.fps
    )


def video_response(
    background_tasks: BackgroundTasks,
    frames: list[Image.Image],        
    fps: float = 24,
    interpolate_film: int = 1,
    interpolate_rife: int = 1,
    fast_interpolate: bool = False,
    audio: str = None,
    return_path=False,
):
    if interpolate_film > 0 or interpolate_rife > 0:
        frames = interpolate_frames(
            frames, interpolate_film, interpolate_rife
        )
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

    for frame in frames:
        writer.append_data(np.array(frame))

    writer.close()

    if audio:
        audio_path = random_filename("wav", False)
        fetch_audio(audio, audio_path)
        add_audio_to_video(full_path, audio_path, full_path)        
        delete_file(audio_path)

    if background_tasks:
        background_tasks.add_task(delete_file, full_path)

    if return_path:
        return full_path

    return FileResponse(
        full_path,
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Length": str(os.path.getsize(full_path))},
    )


def fetch_audio(url: str, save_path: str):
    logging.info(f"Downloading audio from {url}...")
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)


def images_to_arrays(image_objects: list[Image.Image]):
    image_arrays = [np.array(img) for img in image_objects]
    return np.array(image_arrays)


def save_video_from_frames(frames: list, output_path: str, fps: float = 8):
    writer = imageio.get_writer(output_path, format="mp4", fps=fps)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
    return output_path


def frames_to_video(
    video_path, output_path, audio_path=None, audio_url: str = None, fps: float = 24
):
    from moviepy.editor import VideoFileClip, AudioFileClip

    video_clip = VideoFileClip(video_path, fps_source="fps")

    # Set audio if provided
    if audio_path:
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
    elif audio_url:
        # Download audio from URL
        audio_path = random_filename(audio_url.split(".")[-1], True)
        fetch_audio(audio_url, audio_path)
        audio_clip = AudioFileClip(audio_path)
        os.remove(audio_path)
        video_clip = video_clip.set_audio(audio_clip)

    # Write the video file
    video_clip.write_videofile(output_path, codec="libx264", fps=fps)

    return output_path


def interpolate_frames(
    frames: list, interpolate_film: int = 1, interpolate_rife: int = 1
):

    logging.info(f"Interpolating {len(frames)} frames x{interpolate_film}...")

    has_film = os.path.exists("models/film_net/Style/saved_model")

    if interpolate_film > 0:
        if has_film:
            logging.info("Using FiLM model for video frame interpolation.")
            from submodules.frame_interpolation.eval.interpolator import Interpolator

            film_interpolator = (
                Interpolator("models/film_net/Style/saved_model", None)
                if has_film
                else None
            )
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
            frames = [np.array(frame * 255).astype(np.uint8) for frame in frames]

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


def double_frame_rate_with_interpolation(
    input_path, output_path, max_frames: int = None
):
    import cv2

    # Open the video file using imageio
    video_reader = imageio.get_reader(input_path)
    fps = video_reader.get_meta_data()["fps"]

    # Calculate new frame rate
    new_fps = 2 * fps

    metadata = video_reader.get_meta_data()
    print(metadata)

    # Get the video's width and height
    width, height = metadata["size"]
    print(f"Video dimensions: {width}, {height}")

    # Create VideoWriter object to save the output video using imageio
    video_writer = imageio.get_writer(output_path, fps=new_fps)

    # Read the first frame
    prev_frame = video_reader.get_data(0)

    if max_frames is None:
        max_frames = len(video_reader)
    else:
        max_frames = max(max_frames, len(video_reader))

    print(f"Processing {max_frames} frames...")
    for i in range(1, max_frames):
        try:
            # Read the current frame
            frame = video_reader.get_data(i)

            # Linear interpolation between frames
            interpolated_frame = cv2.addWeighted(prev_frame, 0.5, frame, 0.5, 0)

            # Write the original and interpolated frames to the output video
            video_writer.append_data(prev_frame)
            video_writer.append_data(interpolated_frame)

            prev_frame = frame
        except IndexError as e:
            logging.error(f"IndexError: {e}")
            break

    # Close the video writer
    video_writer.close()

    print(f"Video with double frame rate and interpolation saved at: {output_path}")
