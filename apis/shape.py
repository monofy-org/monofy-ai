import logging
from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from utils.file_utils import delete_file, random_filename
from utils.gpu_utils import gpu_thread_lock

router = APIRouter()


@router.get("/shape")
async def shap_e(
    background_tasks: BackgroundTasks,
    prompt: str,
    guidance_scale: float = 15.0,
    format: str = "gif",
    steps: int = 32,
):
    try:
        async with gpu_thread_lock:
            file_path = random_filename()
            from clients import ShapeClient

            ShapeClient.generate(
                prompt,
                file_path,
                guidance_scale=guidance_scale,
                format=format,
                steps=steps,
            )
            file_path = f"{file_path}.{format}"
            background_tasks.add_task(delete_file, file_path)
            if format == "gif":
                media_type = "image/gif"
            else:
                media_type = "application/octet-stream"

            return FileResponse(file_path, media_type=media_type)
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
