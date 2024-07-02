import asyncio
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
        # for i, v in enumerate(voice_manager.voices):
        #     logging.info(f"Voice {i}: {v['ShortName']}")

asyncio.create_task(edge_initialize())

async def generate_speech_edge(
    text: str, voice: str, speed: float = 1.0, pitch: float = 1.0
):
    voices = voice_manager.find(ShortName=voice)
    if len(voices) == 0:
        logging.error(f"Voice {voice} not found.")
        raise HTTPException(status_code=400, detail=f"Voice {voice} not found.")

    print(f"Using voice: {voices[0]['Name']}")

    args = dict({
        "text": text,
        "voice": voices[0]["Name"],        
    })

    if speed != 1:
        prefix = "+" if (speed - 1 >= 0) else ""
        rate = f"{prefix}{round((speed - 1) * 100)}%"
        args["rate"] = rate

    if pitch != 1:
        s = (pitch - 1) * 100
        p = ("+" if s > 0 else "-") + str(s) + "%"
        args["pitch"] = p

    communicate = edge_tts.Communicate(**args)

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data: bytes = chunk["data"]
            yield data


@router.post("/tts/edge", tags=["Text-to-Speech"])
def tts_edge(req: TTSRequest):

    if req.voice == "female1":
        req.voice = "en-US-AvaNeural"

    return StreamingResponse(
        generate_speech_edge(req.text, req.voice, req.speed),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment", "filename": "output.mp3"},
    )


@router.get("/tts/edge/voices", tags=["Text-to-Speech"])
async def tts_edge_voices():
    return voice_manager.voices
