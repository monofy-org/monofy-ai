import logging
import os
import imageio
import numpy as np
from PIL import Image

from utils.file_utils import random_filename


def get_fps(video_path):
    reader = imageio.get_reader(video_path)
    return reader.get_meta_data()["fps"]


def extract_frames(
    video_path, num_frames, trim_start=0, trim_end=0, return_json: bool = False
):
    reader = imageio.get_reader(video_path)
    metadata = reader.get_meta_data()
    duration = metadata.get('duration', 0)
    fps = metadata.get('fps', 30)

    start_time = trim_start
    end_time = duration - trim_end

    frame_duration = (end_time - start_time) / num_frames

    frames = []

    for i in range(num_frames):
        t = (i * frame_duration) + start_time
        frame_index = int(t * fps)
        image = Image.fromarray(reader.get_data(frame_index))
        if return_json:
            frames.append({"time": t, "image": image, "summary": None})
        else:
            frames.append(image)

    reader.close()

    return frames


def remove_audio(path: str, delete_old_file: bool = False):
    import ffmpy

    new_path = random_filename("mp4")
    ff = ffmpy.FFmpeg(inputs={path: None}, outputs={new_path: "-an"})
    ff.run()
    if delete_old_file:
        os.remove(path)
    return new_path


def replace_audio(video_path, audio_path, output_path):
    import ffmpeg

    video = ffmpeg.input(video_path)
    audio = ffmpeg.input(audio_path)

    # Get the duration of the video
    probe = ffmpeg.probe(video_path)
    video_duration = float(probe['streams'][0]['duration'])

    output = ffmpeg.output(
        video.video,
        audio.audio.filter('atrim', duration=video_duration),
        output_path,
        vcodec="copy",
        acodec="aac"
    )

    ffmpeg.run(output, overwrite_output=True)

    return output_path


def fix_video(video_path, delete_old_file: bool = False):
    import ffmpeg

    temp_path = random_filename("mp4")

    input = ffmpeg.input(video_path)
    output = ffmpeg.output(input, temp_path, vcodec="libx264", acodec="aac")
    ffmpeg.run(output, overwrite_output=True)

    if delete_old_file:
        os.remove(video_path)
        os.rename(temp_path, video_path)

    return video_path


def images_to_arrays(image_objects: list[Image.Image]):
    image_arrays = [np.array(img) for img in image_objects]
    return np.array(image_arrays)


def save_video_from_frames(frames: list, output_path: str, fps: float = 8):
    writer = imageio.get_writer(output_path, format="mp4", fps=fps)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
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
