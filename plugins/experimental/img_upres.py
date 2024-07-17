import logging
from typing import Optional
from PIL import Image
from fastapi import Depends
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.image_utils import get_image_from_request, image_to_base64_no_header


class ImgUpresRequest(BaseModel):
    image: str
    return_json: Optional[bool] = False
    scale: Optional[int] = 4


class ImgUpresPlugin(PluginBase):
    name = "Image Upres"
    description = "Image upscale using AuraSR"
    instance = None

    def __init__(self):
        from aura_sr import AuraSR

        super().__init__()

        self.resources["AuraSR"] = AuraSR.from_pretrained("fal-ai/AuraSR")

    async def generate(self, image):
        from aura_sr import AuraSR

        aura_sr: AuraSR = self.resources["AuraSR"]
        return aura_sr.upscale_4x(image)


@PluginBase.router.post("/img/upres", tags=["Image Upscale"])
async def img_upres(req: ImgUpresRequest):
    plugin: ImgUpresPlugin = None
    try:
        plugin: ImgUpresPlugin = await use_plugin(ImgUpresPlugin)
        image = get_image_from_request(req.image)
        output = await plugin.generate(image)

        output = output.resize((image.width * req.scale, image.height * req.scale))

        if req.return_json:
            return {"images": [image_to_base64_no_header(output)]}
        else:
            return output
    except Exception as e:
        logging.error(e, exc_info=True)
    finally:
        if plugin:
            release_plugin(plugin)


@PluginBase.router.get("/img/upres", tags=["Image Upscale"])
async def img_upres_from_url(self, req: ImgUpresRequest = Depends()):
    return await self.img_upres(req.image)
