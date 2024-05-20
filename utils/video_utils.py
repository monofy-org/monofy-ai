import logging
import os
import imageio
import numpy as np
from PIL import Image
from utils.audio_utils import get_audio_from_request
from utils.file_utils import random_filename


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

    audio_clip = AudioFileClip(audio_path)
    video_clip = VideoFileClip(video_path)

    video_clip: VideoFileClip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(output_path, fps=video_clip.fps)


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
        get_audio_from_request(audio_url, audio_path)
        audio_clip = AudioFileClip(audio_path)
        os.remove(audio_path)
        video_clip = video_clip.set_audio(audio_clip)

    # Write the video file
    video_clip.write_videofile(output_path, codec="libx264", fps=fps)

    return output_path


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
