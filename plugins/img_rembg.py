import io
import logging
import cv2
import rembg
import numpy as np
from pydantic import BaseModel
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from modules.plugins import PluginBase, use_plugin
from utils.image_utils import (
    get_image_from_request,
    image_to_base64_no_header,
)


class RembgRequest(BaseModel):
    image: str
    return_json: bool = False


class RembgPlugin(PluginBase):

    name = "Background Remover"
    description = "Remove background from an image"
    instance = None

    def __init__(self):
        super().__init__()

        self.resources["session"] = rembg.new_session()


@PluginBase.router.post("/img/rembg", tags=["Image Processing"])
async def remove_background(req: RembgRequest):
    try:
        plugin: RembgPlugin = await use_plugin(RembgPlugin, True)

        img = get_image_from_request(req.image)

        # img to cv2 imdecode
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Remove background
        result = rembg.remove(img, True, session=plugin.resources["session"])

        # Encode the image to transparent PNG format
        is_success, buffer = cv2.imencode(".png", result)
        io_buf = io.BytesIO(buffer)

        if req.return_json:
            # get PIL Image
            from PIL import Image

            image = Image.open(io_buf)
            image = image.convert("RGBA")
            return {
                "images": [image_to_base64_no_header(image)],
            }

        return StreamingResponse(io_buf, media_type="image/png")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@PluginBase.router.get("/img/rembg", tags=["Image Processing"])
async def remove_background_from_url(req: RembgRequest = Depends()):
    return await remove_background(req)
