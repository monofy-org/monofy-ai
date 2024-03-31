import io
import logging
from tqdm.rich import tqdm
from PIL import Image
from pytubefix import YouTube
from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.file_utils import random_filename
from utils.image_utils import image_to_base64_no_header
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


class YouTubePlugin(PluginBase):

    name = "Tools for YouTube"
    description = "Tools for YouTube, such as analyzing frames and captions"
    instance = None

    def __init__(self):
        # from pytube.innertube import _default_clients
        # _default_clients["ANDROID_EMBED"] = _default_clients["WEB_EMBED"]

        super().__init__()


@PluginBase.router.post("/youtube/download", tags=["YouTube Tools"])
async def download_youtube_video(
    req: YouTubeDownloadRequest,
):
    from pytubefix import YouTube

    yt: YouTube = YouTube(req.url)

    if req.audio_only is True:

        mp3_filename = random_filename("mp3", False)

        print(yt.streams)

        path = (
            yt.streams.filter(only_audio=True)
            .first()
            .download(output_path=".cache", filename=mp3_filename, mp3=True)
        )

        return FileResponse(
            path,
            media_type="audio/mpeg",
            filename=mp3_filename,
        )

    else:

        mp4_filename = random_filename("mp4", False)

        path = (
            yt.streams.filter(progressive=True, file_extension="mp4")
            .order_by("resolution")
            .desc()
            .first()
            .download(output_path=".cache", filename=mp4_filename)
        )

        return FileResponse(
            path,
            media_type="video/mp4",
            filename=mp4_filename,
        )


@PluginBase.router.post("/youtube/captions", tags=["YouTube Tools"])
async def captions(req: YouTubeCaptionsRequest):
    plugin = None

    try:
        from pytubefix import YouTube

        logging.info("Fetching captions for " + req.url)

        # Create a YouTube object
        yt = YouTube(req.url)
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
            summary = await plugin.generate_chat_response(messages=[], context=context)
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

    from pytubefix import YouTube

    yt: YouTube = YouTube(req.url)

    mp4_filename = random_filename("mp4", False)

    path = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
        .download(output_path=".cache", filename=mp4_filename)
    )

    return create_grid(path, req.rows, req.cols)


def create_grid(video_path, rows, cols, width: int = 1280, height: int = 720):

    # create a grid of static images in a single image
    num_frames = rows * cols

    frames = extract_frames(video_path, num_frames, return_json=False)

    width = int(width / cols)
    height = int(height / rows)
    grid_width = cols * width
    grid_height = rows * height
    grid = Image.new("RGB", (grid_width, grid_height))

    for i in range(len(frames)):

        x = i % cols
        y = i // cols

        frame = Image.fromarray(frames[i]).resize((width, height))
        grid.paste(frame, (x * width, y * height))

    output = io.BytesIO()
    grid.save(output, format="PNG")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="image/png",
    )


@PluginBase.router.get("/youtube/grid", tags=["YouTube Tools"])
async def youtube_grid_from_url(
    req: YouTubeGridRequest = Depends(),
):
    return await youtube_grid(req)


@PluginBase.router.post("/youtube/frames", tags=["YouTube Tools"])
async def youtube_frames(req: YouTubeFramesRequest):

    yt: YouTube = YouTube(req.url)

    mp4_filename = random_filename("mp4", False)

    path = (
        yt.streams.filter(progressive=True, file_extension="mp4")
        .order_by("resolution")
        .desc()
        .first()
        .download(output_path=".cache", filename=mp4_filename)
    )

    frames = extract_frames(
        path, req.num_frames, req.trim_start, req.trim_end, return_json=True
    )

    print(frames)

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
