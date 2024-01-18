import logging
import os
from fastapi import HTTPException, BackgroundTasks, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from utils.file_utils import delete_file, random_filename
from utils.gpu_utils import gpu_thread_lock

router = APIRouter()


@router.get("/musicgen")
async def musicgen(
    background_tasks: BackgroundTasks,
    prompt: str,
    duration: float = 10,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    format: str = "wav",
):
    duration = min(duration, 60)
    async with gpu_thread_lock:
        try:
            from clients import MusicGenClient

            file_path_noext = random_filename(None, True)
            file_path = MusicGenClient.generate(
                prompt,
                file_path_noext,
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef,
                format=format,
            )
            background_tasks.add_task(delete_file, file_path)
            return FileResponse(os.path.abspath(file_path), media_type="audio/wav")
        except Exception as e:
            logging.error(e)
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/musicgen/completions")
async def audiogen_completion(
    background_tasks: BackgroundTasks,
    audio: UploadFile,
    prompt: str,
    duration: int = 3,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
):
    try:
        from clients import AudioGenClient

        async with gpu_thread_lock:
            file_path_noext = random_filename(None, True)
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
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
