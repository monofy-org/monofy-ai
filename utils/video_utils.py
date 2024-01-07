import os
import cv2
import imageio
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip
import requests
from utils.file_utils import random_filename


def add_audio_to_video(video_path, audio_path, output_path):
    # Load video clip
    video_clip = VideoFileClip(video_path)

    # Load audio clip
    audio_clip = AudioFileClip(audio_path)

    # Set video clip's audio to the loaded audio clip
    video_clip = video_clip.set_audio(audio_clip)

    # Write the final video with combined audio
    video_clip.write_videofile(
        output_path, codec="libx264", audio_codec="aac", fps=video_clip.fps
    )


def download_audio(url, save_path):
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)


def images_to_arrays(image_objects):
    image_arrays = [np.array(img) for img in image_objects]
    return np.array(image_arrays)


def frames_to_video(
    video_path, output_path, audio_path=None, audio_url: str = None, fps=24
):
    # Create a video clip from the frames array

    video_clip = VideoFileClip(video_path)

    # Set audio if provided
    if audio_path:
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
    elif audio_url:
        # Download audio from URL
        audio_path = random_filename(audio_url.split(".")[-1], True)
        download_audio(audio_url, audio_path)
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        os.remove(audio_path)  # Remove temporary audio file

    # Write the video file
    video_clip.write_videofile(output_path, codec="libx264", fps=fps)


def double_frame_rate_with_interpolation(
    input_path, output_path, max_frames: int = None
):
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
            print(f"IndexError: {e}")
            break

    # Close the video writer
    video_writer.close()

    print(f"Video with double frame rate and interpolation saved at: {output_path}")
