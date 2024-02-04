import asyncio
import logging
import os
from fastapi import HTTPException, BackgroundTasks, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from utils.file_utils import delete_file, random_filename
from utils.gpu_utils import gpu_thread_lock
from clients import MusicGenClient

router = APIRouter()


@router.get("/musicgen")
async def musicgen(
    background_tasks: BackgroundTasks,
    prompt: str,
    duration: float = 10,
    temperature: float = 1.0,
    guidance_scale: float = 3.0,
    format: str = "wav",
    seed: int = -1,
    top_p: float = 0.9,
):
    await asyncio.sleep(0.1)

    duration = min(duration, 60)
    async with gpu_thread_lock:
        try:
            file_path_noext = random_filename()

            musicgen = MusicGenClient.get_instance()

            file_path = musicgen.generate(
                prompt,
                file_path_noext,
                duration=duration,
                temperature=temperature,
                guidance_scale=guidance_scale,
                format=format,
                seed=seed,
                top_p=top_p,
            )

            background_tasks.add_task(delete_file, file_path)

            return FileResponse(os.path.abspath(file_path), media_type="audio/wav")

        except Exception as e:
            logging.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/musicgen/completions")
async def audiogen_completion(
    background_tasks: BackgroundTasks,
    audio: UploadFile,
    prompt: str = "",
    duration: int = 3,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
):
    try:
        from clients import AudioGenClient

        async with gpu_thread_lock:
            file_path_noext = random_filename()
            file_path = AudioGenClient.generate(
                prompt,
                file_path_noext,
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef,
                wav_bytes=await audio.read(),
            )
            background_tasks.add_task(delete_file, file_path)
            return FileResponse(os.path.abspath(file_path), media_type="audio/wav")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
