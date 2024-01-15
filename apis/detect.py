import io
import logging
from fastapi import HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from utils.gpu_utils import gpu_thread_lock

from utils.image_utils import detect_objects

router = APIRouter()


@router.get("/detect")
async def object_detection(background_tasks: BackgroundTasks, image_url: str):
    try:
        async with gpu_thread_lock:
            result_image = detect_objects(image_url, 0.8)
            img_byte_array = io.BytesIO()
            result_image.save(img_byte_array, format="PNG")
            return StreamingResponse(
                io.BytesIO(img_byte_array.getvalue()), media_type="image/png"
            )
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
