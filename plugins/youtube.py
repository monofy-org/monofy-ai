import logging
from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.exllamav2 import ExllamaV2Plugin

class YouTubeCaptionsRequest(BaseModel):
    url: str
    prompt: Optional[str] = "Your task is to give a concise summary (one to 3 sentences) of a YouTube video."
    summary: Optional[bool] = False
    max_response_tokens: Optional[int] = 3000

class YouTubePlugin(PluginBase):

    name = "Tools for YouTube"
    description = "YouTube"
    instance = None

    def __init__(self):
        # from pytube.innertube import _default_clients
        # _default_clients["ANDROID_EMBED"] = _default_clients["WEB_EMBED"]

        super().__init__()


@PluginBase.router.post("/youtube/captions")
async def captions(
    req: YouTubeCaptionsRequest
):
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
            plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin)
            context = req.prompt + "\nHere is the closed caption transcription:\n\n"
            summary = plugin.generate_chat_response(
                text=text, messages=[], context=context
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

@PluginBase.router.get("/youtube/captions")
async def captions_from_url(
    req: YouTubeCaptionsRequest = Depends(),
):
    return await captions(req)