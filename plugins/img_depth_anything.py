import logging
import os
import numpy as np
from PIL import Image
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from modules.plugins import PluginBase, use_plugin
from settings import CACHE_PATH
from utils.image_utils import (
    get_image_from_request,
    image_to_base64_no_header,
    image_to_bytes,
)


class DepthRequest(BaseModel):
    image: str
    median_filter: int = 5
    return_json: bool = False


class DepthAnythingPlugin(PluginBase):

    from torch import Tensor

    name = "Depth estimation (DepthAnything)"
    description = (
        "Depth estimation using DepthAnything model. Returns a depth map image."
    )
    instance = None

    def __init__(self):
        from transformers import Pipeline, pipeline

        super().__init__()
        pipe: Pipeline = pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=self.device,
        )
        self.resources["DepthAnything"] = pipe

    async def generate_depthmap(
        self, image: Image.Image | Tensor, median_filter: int = 5
    ):

        from transformers import Pipeline
        from scipy.signal import medfilt

        pipe: Pipeline = self.resources["DepthAnything"]
        depth: Image.Image = pipe(image)["depth"]

        if median_filter > 0:
            depth = np.array(depth)
            depth = medfilt(depth, 5)
            depth = Image.fromarray(depth)

        return depth


@PluginBase.router.post("/img/depth", tags=["Image Processing"])
async def depth_estimation(
    req: DepthRequest,
):
    try:
        plugin: DepthAnythingPlugin = await use_plugin(DepthAnythingPlugin, True)
        img = get_image_from_request(req.image)
        depth: Image.Image = await plugin.generate_depthmap(img)
        depth.resize((img.width, img.height), Image.BICUBIC)

        print(f"Depth shape: {np.array(depth).shape}")

        if req.return_json:
            return {
                "images": [image_to_base64_no_header(depth)],
                "media_type": "image/png",
            }
        else:
            return StreamingResponse(image_to_bytes(depth), media_type="image/png")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@PluginBase.router.get("/img/depth", tags=["Image Processing"])
async def depth_estimation_from_url(
    req: DepthRequest = Depends(),
):
    return await depth_estimation(req)
