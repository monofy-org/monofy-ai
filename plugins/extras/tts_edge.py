import io
import logging
from typing import Any
import edge_tts
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from modules.plugins import router
from plugins.tts import TTSRequest
from utils.text_utils import process_llm_text
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
    if voice_manager is None:
        await edge_initialize()
    voices = voice_manager.find(ShortName=req.voice)
    if len(voices) == 0:
        logging.error(f"Voice {req.voice} not found.")
        raise HTTPException(status_code=400, detail=f"Voice {req.voice} not found.")

    print(f"Using voice: {voices[0]['Name']}")
    communicate = edge_tts.Communicate(process_llm_text(req.text), voices[0]["Name"])

    result = io.BytesIO()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            result.write(chunk["data"])
    result.seek(0)

    return StreamingResponse(
        result,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment", "filename": "output.mp3"},
    )


@router.get("/tts/edge/voices", tags=["Text-to-Speech"])
async def tts_edge_voices():
    if voice_manager is None:
        await edge_initialize()
    return voice_manager.voices
