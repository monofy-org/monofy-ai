import io
import logging
from typing import Optional
import numpy as np
from PIL import Image
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from modules.plugins import router
from utils.image_utils import (
    get_image_from_request,
    image_to_base64_no_header,
    image_to_bytes,
)


class CannyRequest(BaseModel):
    image: str
    threshold1: Optional[int] = 100
    threshold2: Optional[int] = 200
    width: Optional[int] = None
    height: Optional[int] = None
    return_json: Optional[bool] = False


def canny_outline(
    image: str,
    threshold1: Optional[int] = 100,
    threshold2: Optional[int] = 200,
    width: Optional[int] = None,
    height: Optional[int] = None,    
):
    import cv2

    crop = (width, height) if width and height else None
    image = get_image_from_request(image, crop)
    image = np.array(image)
    outline = cv2.Canny(image, threshold1, threshold2)
    outline_color = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
    is_success, buffer = cv2.imencode(".png", outline_color)

    if not is_success:
        raise HTTPException(status_code=500, detail="Error encoding image")

    io_buf = io.BytesIO(buffer)
    return Image.open(io_buf)


@router.post("/img/canny", tags=["Image Processing"])
async def canny(
    req: CannyRequest,
):
    try:
        img = canny_outline(req.image, req.threshold1, req.threshold2, req.width, req.height)

        if req.return_json:
            return {
                "images": [image_to_base64_no_header(img)],
                "media_type": "image/png",
            }
        else:
            return StreamingResponse(image_to_bytes(img), media_type="image/png")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/img/canny", tags=["Image Processing"])
async def canny_from_url(
    req: CannyRequest = Depends(),
):
    return await canny(req)
