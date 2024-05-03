import logging
from typing import Any
import edge_tts
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from modules.plugins import router
from plugins.tts import TTSRequest
from utils.text_utils import process_text_for_tts
from edge_tts import VoicesManager

voice_manager: VoicesManager = None
voices: list[str, Any] = None


async def edge_initialize():
    global voice_manager
    global voices
    if voice_manager is None:
        voice_manager = await VoicesManager.create()
        logging.info("Got edge voices.")
        for i, v in enumerate(voice_manager.voices):
            logging.info(f"Voice {i}: {v['ShortName']}")


@router.post("/tts/edge", tags=["Text-to-Speech"])
async def tts_edge(req: TTSRequest):

    result_io = await generate_speech_edge(req.text, req.voice, req.speed)
    return StreamingResponse(
        result_io,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment", "filename": "output.mp3"},
    )


async def generate_speech_edge(text: str, voice: str, speed: float = 1.0):
    if voice_manager is None:
        await edge_initialize()
    voices = voice_manager.find(ShortName=voice)
    if len(voices) == 0:
        logging.error(f"Voice {voice} not found.")
        raise HTTPException(status_code=400, detail=f"Voice {voice} not found.")

    print(f"Using voice: {voices[0]['Name']}")
    prefix = "+" if (speed - 1 >= 0) else ""
    rate = f"{prefix}{round((speed - 1) * 100)}%"
    communicate = edge_tts.Communicate(
        process_text_for_tts(text), voices[0]["Name"], rate=rate
    )

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data: bytes = chunk["data"]
            yield data


@router.get("/tts/edge/voices", tags=["Text-to-Speech"])
async def tts_edge_voices():
    if voice_manager is None:
        await edge_initialize()
    return voice_manager.voices
