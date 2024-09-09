import logging
import os
import imageio
import numpy as np
import imageio_ffmpeg as ffmpeg
from PIL import Image


from utils.file_utils import download_to_cache, random_filename


def get_fps(video_path):
    reader = imageio.get_reader(video_path)
    return reader.get_meta_data()["fps"]


def extract_frames(
    video_path, num_frames, trim_start=0, trim_end=0, return_json: bool = False
):
    reader = imageio.get_reader(video_path)
    metadata = reader.get_meta_data()
    duration = metadata.get("duration", 0)
    fps = metadata.get("fps", 30)

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

    video = ffmpeg.input(video_path)
    audio = ffmpeg.input(audio_path)

    # Get the duration of the video
    probe = ffmpeg.probe(video_path)
    video_duration = float(probe["streams"][0]["duration"])

    output = ffmpeg.output(
        video.video,
        audio.audio.filter("atrim", duration=video_duration),
        output_path,
        vcodec="copy",
        acodec="aac",
    )

    ffmpeg.run(output, overwrite_output=True)

    return output_path


def fix_video(video_path, delete_old_file: bool = False):
    import imageio

    temp_path = random_filename("mp4")

    try:
        reader = imageio.get_reader(video_path)
        writer = imageio.get_writer(temp_path, fps=reader.get_meta_data()["fps"])

        for frame in reader:
            writer.append_data(frame)

        reader.close()
        writer.close()
    except Exception as e:
        logging.error("Error while fixing video", exc_info=True)

    if delete_old_file:
        os.remove(video_path)
        os.rename(temp_path, video_path)
        return video_path
    else:
        return temp_path


def save_video_from_frames(frames: list, output_path: str, fps: float = 8):
    writer = imageio.get_writer(output_path, format="mp4", fps=fps)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()
    return output_path


async def get_video_from_request(video: str) -> str:
    is_url = "://" in video

    if is_url:
        if "youtube.com" in video or "youtu.be" in video:
            import plugins.extras.youtube

            return plugins.extras.youtube.download(video)
        elif "reddit.com" in video:
            import plugins.extras.reddit

            return plugins.extras.reddit.download_to_cache(video)
        else:
            return download_to_cache(video)
    else:
        extension = video.lower().split(".")[-1]
        if extension in ["mp4", "mov", "avi", "webm", "wmv", "flv", "mkv"]:
            return video
        else:
            raise ValueError("Invalid video format")
