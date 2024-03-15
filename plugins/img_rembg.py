import io
import logging
import cv2
import rembg
import numpy as np
from pydantic import BaseModel
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from modules.plugins import PluginBase, use_plugin
from utils.image_utils import crop_and_resize, get_image_from_request


class RembgRequest(BaseModel):
    image: str
    width: int
    height: int


class RembgPlugin(PluginBase):

    name = "Rembg"
    description = "Remove background from an image"
    instance = None

    def __init__(self):
        super().__init__()

        self.resources["rembg"] = rembg.new_session()


@PluginBase.router.post("/rembg")
async def remove_background(req: RembgRequest):
    try:
        plugin: RembgPlugin = await use_plugin(RembgPlugin, True)

        img = get_image_from_request(req.image)

        if req.width and req.height:
            img = crop_and_resize(img, (req.width, req.height))

        # img to cv2 imdecode
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Remove background
        result = rembg.remove(img, True, session=plugin.resources["rembg"])

        # Encode the image to transparent PNG format
        is_success, buffer = cv2.imencode(".png", result)
        io_buf = io.BytesIO(buffer)

        return StreamingResponse(io_buf, media_type="image/png")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
