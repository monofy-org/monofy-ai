import io
import logging
import time
from typing import Literal, Optional

from imageio import mimwrite
from moviepy.editor import VideoFileClip
import numpy as np
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from pytubefix import YouTube
from tqdm.rich import tqdm

from modules.plugins import PluginBase, release_plugin, use_plugin
from settings import CACHE_PATH
from utils.file_utils import random_filename, url_hash
from utils.image_utils import image_to_base64_no_header, image_to_bytes
from utils.video_utils import extract_frames


class YouTubeCaptionsRequest(BaseModel):
    url: str
    prompt: Optional[str] = (
        "Your task is to give a concise summary (one to 3 sentences) of a YouTube video."
    )
    summary: Optional[bool] = False
    max_response_tokens: Optional[int] = 3000


class YouTubeDownloadRequest(BaseModel):
    url: str
    audio_only: Optional[bool] = False
    start_time: Optional[int] = 0
    length: Optional[float] = None
    format: Optional[Literal["mp4", "gif"]] = "mp4"
    fps: Optional[int] = 10
    text: Optional[str] = None
    width: Optional[int] = None


class YouTubeGridRequest(BaseModel):
    url: str
    rows: int = 3
    cols: int = 3


class YouTubeFramesRequest(BaseModel):
    url: str
    num_frames: Optional[int] = 10
    trim_start: Optional[int] = 2
    trim_end: Optional[int] = 2
    summary: Optional[bool] = False
    captions: Optional[bool] = False


@PluginBase.router.post("/youtube/download", tags=["YouTube Tools"])
async def download_youtube_video(
    req: YouTubeDownloadRequest,
):
    path = download_media(**req.__dict__)
    return FileResponse(
        path, media_type="image/gif" if format == "gif" else "video/mp4"
    )


def download_media(
    url: str,
    start_time: Optional[float] = 0,
    length: Optional[float] = None,
    audio_only: Optional[bool] = False,
    format: Optional[Literal["mp4", "gif"]] = "mp4",
    fps: Optional[int] = 10,
    text: Optional[str] = None,
    width: Optional[int] = None,
    filename: Optional[str] = None,
):
    filename = filename or f"{url_hash(url)}.{format}"

    yt: YouTube = YouTube(url, "WEB")

    # extract start time from url
    start_time_seconds = 0

    if format == "gif":
        if length and length > 3:
            raise HTTPException(
                status_code=400,
                detail="GIF length cannot exceed 3 seconds",
            )

    if "t=" in url:
        start_time = url.split("t=")[1]
        if "&" in start_time:
            start_time = start_time.split("&")[0]
        if "h" in start_time:
            start_time_seconds += int(start_time.split("h")[0]) * 3600
            start_time = start_time.split("h")[1]
        if "m" in start_time:
            start_time_seconds += int(start_time.split("m")[0]) * 60
            start_time = start_time.split("m")[1]
        if "s" in start_time:
            start_time_seconds += float(start_time.split("s")[0])

        if start_time_seconds == 0 and float(start_time) > 0:
            start_time_seconds = float(start_time)

        start_time = start_time_seconds

    if audio_only is True:
        # print(yt.streams)
        path = yt.streams.get_audio_only().download(
            output_path=CACHE_PATH, filename=filename
        )

        return path

    else:
        path = yt.streams.get_highest_resolution().download(
            output_path=CACHE_PATH, filename=filename
        )

        # trim video to start and end time
        end_time = start_time_seconds + length if length is not None else None

        clip: VideoFileClip = VideoFileClip(path)

        if (start_time > 0) or end_time is not None:
            clip = clip.subclip(start_time, end_time)

        if format == "mp4":
            if start_time > 0 or end_time is not None:
                path = path.replace(".mp4", "_trimmed.mp4")
                clip.write_videofile(path)

            clip.close()
            return path

        elif format == "gif":
            path = path.replace(".mp4", ".gif")

            if text:
                # we don't have ImageMagick so we can't use TextClip
                frames = []
                for frame in clip.iter_frames(fps=fps):
                    img: Image.Image = Image.fromarray(frame)
                    d = ImageDraw.Draw(img)
                    font = ImageFont.truetype("impact.ttf", 72)  # use Impact font
                    size = font.getbbox(text)
                    text_x = (img.width - size[2]) / 2
                    text_y = img.height - size[3] - 30
                    d.text(
                        (text_x, text_y),
                        text,
                        fill="white",
                        font=font,
                        align="center",
                    )
                    if width is not None:
                        ar = img.width / img.height
                        img = img.resize((width, int(width / ar)))

                    frames.append(img)

            # convert frames to gif
            mimwrite(path, frames, format="gif", fps=fps, loop=0)

            return path
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid format",
            )


@PluginBase.router.post("/youtube/captions", tags=["YouTube Tools"])
async def captions(req: YouTubeCaptionsRequest):
    plugin = None

    try:
        logging.info("Fetching captions for " + req.url)

        # Create a YouTube object
        yt = YouTube(req.url, "WEB")
        # try:
        #     yt.bypass_age_gate()
        # except Exception as e:
        #     logging.warning("Failed to bypass age gate: " + str(e))
        #     yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
        #     yt.bypass_age_gate()

        # tracks = yt.caption_tracks

        caption = yt.captions["a.en"]

        caption = caption.json_captions

        text = ""
        tokens = 0

        for item in caption["events"]:
            if "segs" in item:
                for seg in item["segs"]:
                    text += seg["utf8"]
                    tokens += 1
                    if tokens > req.max_response_tokens:
                        logging.warning(
                            "Exceeded max tokens, summary may be incomplete"
                        )
                        break
                if tokens > req.max_response_tokens:
                    break

        if req.summary:
            from plugins.exllamav2 import ExllamaV2Plugin

            plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin)
            context = (
                req.prompt
                + "\nHere is the closed caption transcription:\n\n"
                + text
                + "\n\nGive your response now:\n\n"
            )
            print(context)
            summary = "".join(
                [
                    x
                    async for x in plugin.generate_chat_response(
                        messages=[], context=context
                    )
                ]
            )
            return {
                "captions": text,
                "summary": summary,
            }

        return JSONResponse(content={"captions": text})

    except Exception as e:
        logging.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if plugin:
            release_plugin(ExllamaV2Plugin)


@PluginBase.router.get("/youtube/captions", tags=["YouTube Tools"])
async def captions_from_url(
    req: YouTubeCaptionsRequest = Depends(),
):
    return await captions(req)


@PluginBase.router.get("/youtube/download", tags=["YouTube Tools"])
async def download_youtube_video_from_url(
    req: YouTubeDownloadRequest = Depends(),
):
    return await download_youtube_video(req)


@PluginBase.router.post("/youtube/grid", tags=["YouTube Tools"])
async def youtube_grid(req: YouTubeGridRequest):
    logging.info("Creating grid for " + req.url)

    start_time = time.time()

    yt: YouTube = YouTube(req.url, "WEB")

    mp4_filename = random_filename("mp4", False)

    path = yt.streams.get_highest_resolution().download(
        output_path=CACHE_PATH, filename=mp4_filename
    )

    grid = create_grid(path, req.rows, req.cols, 2, 2)

    duration = time.time() - start_time
    logging.info("Grid created in " + str(duration) + " seconds")

    return StreamingResponse(image_to_bytes(grid), media_type="image/png")


def create_grid(
    video_path,
    rows,
    cols,
    scale: float = 0.25,
    trim_start=0,
    trim_end=0,
):
    # create a grid of static images in a single image
    num_frames = rows * cols

    frames: list[Image.Image] = extract_frames(
        video_path,
        num_frames,
        return_json=False,
        trim_start=trim_start,
        trim_end=trim_end,
    )

    width = int(frames[0].width * scale)
    height = int(frames[0].height * scale)
    grid_width = cols * width
    grid_height = rows * height
    grid = Image.new("RGB", (grid_width, grid_height))

    for i in range(len(frames)):
        x = i % cols
        y = i // cols

        frame = frames[i].resize((width, height))
        grid.paste(frame, (x * width, y * height))

    output = io.BytesIO()
    grid.save(output, format="PNG")
    return grid


@PluginBase.router.get("/youtube/grid", tags=["YouTube Tools"])
async def youtube_grid_from_url(
    req: YouTubeGridRequest = Depends(),
):
    return await youtube_grid(req)


@PluginBase.router.post("/youtube/frames", tags=["YouTube Tools"])
async def youtube_frames(req: YouTubeFramesRequest):
    yt: YouTube = YouTube(req.url, "WEB")

    mp4_filename = random_filename("mp4", False)

    path = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
        .download(output_path=CACHE_PATH, filename=mp4_filename)
    )

    frames = extract_frames(
        path,
        req.num_frames,
        req.trim_start,
        req.trim_end,
        return_json=True,
        trim_start=2,
        trim_end=2,
    )

    if req.summary:
        from plugins.img2txt_moondream import Img2TxtMoondreamPlugin

        plugin: Img2TxtMoondreamPlugin = await use_plugin(Img2TxtMoondreamPlugin)

        try:
            # show tqdm progress bar
            with tqdm(
                total=len(frames), unit="frame", desc="Getting descriptions..."
            ) as pbar:
                for frame in frames:
                    summary = await plugin.generate_response(
                        frame["image"],
                        "Describe this image in complete detail including people, background, objects, actions, etc.",
                    )
                    frame["image"] = image_to_base64_no_header(frame["image"])
                    frame["summary"] = summary
                    pbar.update(1)
        except Exception as e:
            logging.error(str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if plugin:
                release_plugin(Img2TxtMoondreamPlugin)

    else:
        for frame in frames:
            frame["image"] = image_to_base64_no_header(frame["image"])

    return {
        "title": yt.title,
        "length": yt.length,
        "frames": frames,
    }


@PluginBase.router.get("/youtube/frames", tags=["YouTube Tools"])
async def youtube_frames_from_url(
    req: YouTubeFramesRequest = Depends(),
):
    return await youtube_frames(req)
