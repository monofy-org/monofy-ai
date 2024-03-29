import asyncio
import logging
import os
from fastapi import BackgroundTasks, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from utils.gpu_utils import gpu_thread_lock, load_gpu_task
from utils.file_utils import delete_file, random_filename

router = APIRouter()


@router.get("/audiogen")
async def audiogen(
    background_tasks: BackgroundTasks,
    prompt: str,
    duration: int = 3,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
):
    try:
        from clients import AudioGenClient

        await asyncio.sleep(0.1)

        async with load_gpu_task("audiogen", AudioGenClient):
            file_path_noext = random_filename()
            file_path = AudioGenClient.generate(
                prompt,
                file_path_noext,
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef,
            )
            background_tasks.add_task(delete_file, file_path)
            return FileResponse(os.path.abspath(file_path), media_type="audio/wav")
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audiogen/completions")
async def audiogen_completion(
    background_tasks: BackgroundTasks,
    audio: UploadFile,
    prompt: str = "",
    duration: int = 3,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    top_p: float = 1.0,
):
    try:
        from clients import AudioGenClient

        async with load_gpu_task("audiogen", AudioGenClient):
            file_path_noext = random_filename()
            file_path = AudioGenClient.generate(
                prompt,
                file_path_noext,
                duration=duration,
                temperature=temperature,
                cfg_coef=cfg_coef,
                top_p=top_p,
                wav_bytes=await audio.read(),
            )
            background_tasks.add_task(delete_file, file_path)
            return FileResponse(os.path.abspath(file_path), media_type="audio/wav")
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
