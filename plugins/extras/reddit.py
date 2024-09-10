import io
import logging
from typing import Optional
from fastapi import Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from modules.plugins import router
import requests
import re
import imageio_ffmpeg as ffmpeg
import subprocess
import os
import bs4

from settings import CACHE_PATH
from utils.console_logging import log_recycle
from utils.file_utils import random_filename, url_hash


class RedditDownloadRequest(BaseModel):
    url: str
    audio_only: Optional[bool] = False


def get_playlists(m3u8_url: str) -> tuple[str, str]:

    logging.info("Getting playlists from " + m3u8_url)

    # Fetch the M3U8 file content
    response = requests.get(m3u8_url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch main playlist ({response.status_code}): {m3u8_url}"
        )

    base_url = m3u8_url.rsplit("/", 1)[0]

    # Split the content by newline
    lines = response.text.split("\n")

    max_video_bandwidth = 0
    max_audio_quality = -1
    audio_playlist = None
    video_playlist = None

    # Loop through the lines
    for i, line in enumerate(lines):

        if not line:
            continue

        data = line.split(",")
        properties = {}
        for prop in data:
            s = prop.split("=", 1)
            if len(s) == 2:
                key, value = s
                key = key.split(":")[-1]
                properties[key] = value.strip('"')

        uri = properties.get("URI")
        bandwidth = properties.get("BANDWIDTH")

        if bandwidth:
            try:
                video_bandwidth = int(bandwidth)

                if video_bandwidth > max_video_bandwidth:
                    max_video_bandwidth = video_bandwidth
                    video_playlist = f"{base_url}/{uri or lines[i+1]}"
            except (IndexError, ValueError):
                continue

        # Find the highest audio number
        elif uri:
            audio_quality = int(re.search(r"HLS_AUDIO_(\d+)", line).group(1))

            if audio_quality > max_audio_quality:
                max_audio_quality = audio_quality
                audio_playlist = f"{base_url}/{uri}"

    if not video_playlist:
        raise Exception("No video playlist found")
    if not audio_playlist:
        raise Exception("No audio playlist found")

    return video_playlist, audio_playlist


def _hls_download(playlist_url: str):
    response = requests.get(playlist_url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch audio playlist ({response.status_code}): {playlist_url}"
        )

    lines = response.text.split("\n")

    base_url = playlist_url.rsplit("/", 1)[0]

    for line in lines:
        if line.startswith("HLS_"):
            media_url = f"{base_url}/{line.strip()}"
            break

    if not media_url:
        raise Exception("Failed to find audio filename")

    logging.info(f"Fetching media from {media_url}")

    response = requests.get(media_url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch audio file ({response.status_code}): {media_url}"
        )

    return response.content


def download_from_playlist(m3u8_stream_url: str, audio_only: bool = False) -> bytes:

    video_playlist, audio_playlist = get_playlists(m3u8_stream_url)

    audio_content = _hls_download(audio_playlist)

    if audio_only:
        return audio_content

    video_content = _hls_download(video_playlist)

    video_file = random_filename("ts")
    audio_file = random_filename("aac")
    output_file = f"{CACHE_PATH}/{url_hash(m3u8_stream_url)}.mp4"

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "rb") as f:
            log_recycle(f"Using cached video {output_file}")
            return f.read()

    with open(video_file, "wb") as f:
        f.write(video_content)

    with open(audio_file, "wb") as f:
        f.write(audio_content)

    # Combine video and audio using ffmpeg
    command = [
        ffmpeg.get_ffmpeg_exe(),
        "-i",
        video_file,
        "-i",
        audio_file,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        "-y",
        output_file,
    ]

    subprocess.run(command, check=True)

    with open(output_file, "rb") as f:
        combined_content = f.read()

    # Clean up temporary files
    os.remove(video_file)
    os.remove(audio_file)
    os.remove(output_file)

    return combined_content


def download_video_bytes(url: str, audio_only: bool = False) -> bytes:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the file: {response.status_code}")

    soup = bs4.BeautifulSoup(response.text, "html.parser")
    source = soup.find("source")
    src = source.get("src")
    if src is None:
        raise Exception("No video source found on reddit link")

    return download_from_playlist(src, audio_only)


def download_to_cache(url: str, audio_only: bool = False, filename: str = None) -> str:
    content = download_video_bytes(url, audio_only)

    if not content:
        raise Exception("Failed to download content")

    filename = filename or random_filename("mp4")
    with open(filename, "wb") as f:
        f.write(content)
        return filename


@router.post("/reddit/download")
async def download(req: RedditDownloadRequest):

    content = download_video_bytes(**req.__dict__)

    return StreamingResponse(
        io.BytesIO(content), media_type="audio/mpeg" if req.audio_only else "video/mp4"
    )


@router.get("/reddit/download_from_url")
async def download_from_url(req: RedditDownloadRequest = Depends()):
    return await download(req)
