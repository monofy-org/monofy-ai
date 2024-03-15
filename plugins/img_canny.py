import io
import logging
from typing import Optional
import numpy as np
from PIL import Image
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
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


class CannyPlugin(PluginBase):

    name = "Canny"
    description = "Canny edge detection"
    instance = None

    async def generate(
        self,
        req: CannyRequest,
    ):
        import cv2

        img = get_image_from_request(req.image, (req.width, req.height))
        img = np.array(img)
        outline = cv2.Canny(img, req.threshold1, req.threshold2)
        outline_color = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
        is_success, buffer = cv2.imencode(".png", outline_color)

        if not is_success:
            raise HTTPException(status_code=500, detail="Error encoding image")

        io_buf = io.BytesIO(buffer)
        return Image.open(io_buf)


@PluginBase.router.post(
    "/img/canny", tags=["Image Processing"]
)
async def canny(
    req: CannyRequest,
):
    try:
        plugin: CannyPlugin = await use_plugin(CannyPlugin, True)
        img = await plugin.generate(req)

        if req.return_json:
            return {
                "image": image_to_base64_no_header(img),
                "media_type": "image/png",
            }
        else:
            return StreamingResponse(image_to_bytes(img), media_type="image/png")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@PluginBase.router.get(
    "/img/canny", tags=["Image Processing"]
)
async def canny_from_url(
    req: CannyRequest = Depends(),
):
    plugin = None
    
    try:
        plugin: CannyPlugin = await use_plugin(CannyPlugin, True)
        img = await plugin.generate(req)

        return StreamingResponse(image_to_bytes(img), media_type="image/png")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if plugin:
            release_plugin(CannyPlugin)
