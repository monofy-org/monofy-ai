import asyncio
import io
import logging
import time
import soundfile as sf
from fastapi import HTTPException, BackgroundTasks, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
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
    streaming: bool = False,
):
    await asyncio.sleep(0.1)
    time.sleep(0.1)

    duration = 30 if duration > 30 else duration

    try:
        chunks = MusicGenClient.get_instance().generate(
            prompt,
            duration=duration,
            temperature=temperature,
            guidance_scale=guidance_scale,
            format=format,
            seed=seed,
            top_p=top_p,
            streaming=streaming,
        )
        
        async for sampling_rate, chunk in chunks:
            bytes_io = io.BytesIO()
            chunk = chunk.reshape(-1, 1)
            sf.write(bytes_io, chunk, sampling_rate, format="wav")
            bytes_io.seek(0)
            return StreamingResponse(bytes_io, media_type="audio/wav")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    #return StreamingResponse(io.BytesIO(wav_output), media_type="audio/wav")


@router.post("/musicgen/completions")
async def musicgen_completion(
    background_tasks: BackgroundTasks,
    audio: UploadFile,
    prompt: str = "",
    duration: int = 30,
    temperature: float = 0.9,
    guidance_scale: float = 5.0,
    format: str = "wav",
    seed: int = -1,
    top_k: int = 250,
    top_p: float = 0.95,    
):
    try:

        sampling_rate, wav_output = MusicGenClient.get_instance().generate(
            prompt,
            duration=duration,
            temperature=temperature,
            guidance_scale=guidance_scale,
            format=format,
            seed=seed,
            #top_k=top_k,
            top_p=top_p,
            wav_bytes=await audio.read(),
        )

        return StreamingResponse(io.BytesIO(wav_output), media_type="audio/wav")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
