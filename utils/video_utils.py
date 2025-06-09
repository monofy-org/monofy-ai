import logging
import os

import imageio
from PIL import Image

from utils.file_utils import (
    download_to_cache,
    get_cached_media,
    random_filename,
)


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
    import ffmpy

    video = ffmpy.FFmpeg(
        inputs={video_path: None, audio_path: None},
        outputs={output_path: "-c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0"},
    )

    video.run()
    return output_path

def fix_video(video_path, delete_old_file: bool = False, crop_and_resize: tuple = None):
    import imageio

    temp_path = random_filename("mp4")

    try:
        reader = imageio.get_reader(video_path)
        writer = imageio.get_writer(temp_path, fps=reader.get_meta_data()["fps"])

        for frame in reader:
            if crop_and_resize:
                target_width, target_height = crop_and_resize
                frame_height, frame_width = frame.shape[:2]

                # Calculate scaling factor to maintain aspect ratio
                scale = max(target_width / frame_width, target_height / frame_height)

                # Calculate new dimensions
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)

                # Resize frame
                frame = imageio.v3.imresize(frame, (new_height, new_width))

                # Calculate crop coordinates to center the frame
                start_x = (new_width - target_width) // 2
                start_y = (new_height - target_height) // 2

                # Crop frame
                frame = frame[
                    start_y : start_y + target_height, start_x : start_x + target_width
                ]

            writer.append_data(frame)

        reader.close()
        writer.close()
    except Exception:
        logging.error("Error while fixing video", exc_info=True)

    if delete_old_file:
        os.remove(video_path)
        os.rename(temp_path, video_path)
        return video_path
    else:
        return temp_path


def get_video_from_request(url_or_path: str, audio_only=False) -> str:
    cached_filename = get_cached_media(url_or_path, audio_only)
    if cached_filename:
        from utils.console_logging import log_recycle

        log_recycle(f"Using cached file: {cached_filename}")
        return cached_filename

    is_url = "://" in url_or_path

    if is_url:
        from urllib.parse import urlparse

        parsed_url = urlparse(url_or_path)
        domain = parsed_url.netloc.split(".", 1)[-1]

        if domain in ["youtube.com", "youtu.be"]:
            import plugins.extras.youtube

            return plugins.extras.youtube.download_media(
                url_or_path, audio_only=audio_only
            )
        elif domain in ["reddit.com", "redd.it"]:
            import plugins.extras.reddit

            return plugins.extras.reddit.download_media(
                url_or_path, audio_only=audio_only
            )
        elif url_or_path.endswith(".ts"):
            from ffmpy import FFmpeg

            filename = download_to_cache(url_or_path, "ts")
            outfile = random_filename("mp4")

            ff = FFmpeg(inputs={filename: None}, outputs={outfile: "-c copy"})
            ff.run()
            os.remove(filename)
            return outfile
        else:
            return download_to_cache(url_or_path, "mp4")
    else:
        extension = url_or_path.lower().split(".")[-1]
        if extension in ["mp4", "mov", "avi", "webm", "wmv", "flv", "mkv", "ts"]:
            return url_or_path
        else:
            raise ValueError("Invalid video format")
