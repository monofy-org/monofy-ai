import asyncio
import io
import logging
from fastapi import HTTPException, BackgroundTasks, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from utils.gpu_utils import gpu_thread_lock
from clients.MusicGenClient import MusicGenClient

router = APIRouter()


@router.get("/musicgen")
async def musicgen(
    background_tasks: BackgroundTasks,
    prompt: str,
    duration: float = 10,
    temperature: float = 1.05,
    guidance_scale: float = 3.0,
    format: str = "wav",
    seed: int = -1,
    top_p: float = 0.95,
):
    await asyncio.sleep(0.1)

    duration = 60 if duration > 60 else duration

    async with gpu_thread_lock:
        try:

            wav_output = MusicGenClient.get_instance().generate(
                prompt,
                duration=duration,
                temperature=temperature,
                guidance_scale=guidance_scale,
                format=format,
                seed=seed,
                top_p=top_p,
            )

            return StreamingResponse(io.BytesIO(wav_output), media_type="audio/wav")

        except Exception as e:
            logging.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/musicgen/completions")
async def musicgen_completion(
    background_tasks: BackgroundTasks,
    audio: UploadFile,
    prompt: str = "",
    duration: int = 30,
    temperature: float = 1.0,
    guidance_scale: float = 3.0,
    format: str = "wav",
    seed: int = -1,
    top_p: float = 0.9,
):
    try:        

        wav_output = MusicGenClient.get_instance().generate(
            prompt,            
            duration=duration,
            temperature=temperature,
            guidance_scale=guidance_scale,
            format=format,
            seed=seed,
            top_p=top_p,
            wav_bytes=await audio.read(),
        )

        return StreamingResponse(io.BytesIO(wav_output), media_type="audio/wav")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
