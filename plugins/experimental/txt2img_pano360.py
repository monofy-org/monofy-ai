import logging
import os
import torch
from typing import Optional
from fastapi import Depends, HTTPException
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.stable_diffusion import format_response
from submodules.Pano360.txt2panoimg.text_to_360panorama_image_pipeline import (
    Text2360PanoramaImagePipeline,
)
from utils.file_utils import cached_snapshot


class Txt2ImgPano360Request(BaseModel):
    prompt: str
    refinement: Optional[bool] = True
    upscale: Optional[bool] = False


class Txt2ImgPano360Plugin(PluginBase):
    name = "Text-to-Image (Pano360)"
    description = "Text-to-Image using Pano360"
    instance = None

    def __init__(self):
        super().__init__()

        self.dtype = torch.float16

        model_path = cached_snapshot("archerfmy0831/sd-t2i-360panoimage")
        model_path = os.path.join(model_path, "sr-base")

        try:
            self.resources["pipeline"] = Text2360PanoramaImagePipeline.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device=self.device,
            ).to(self.device)
        except Exception as e:
            logging.error(e, exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def generate(self, prompt: str, **kwargs):

        logging.info(f"txt2img_pano360: {prompt}")

        args = dict(prompt=prompt, refinement=True, upscale=False)
        args.update(kwargs)

        pipeline = self.resources["pipeline"]
        return pipeline(**args)


@PluginBase.router.post("/txt2img/pano360")
async def txt2img_pano360(request: Txt2ImgPano360Request):
    plugin: Txt2ImgPano360Plugin = None
    try:
        plugin = await use_plugin(Txt2ImgPano360Plugin)
        image = await plugin.generate(request)

        return format_response(image)

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            release_plugin(plugin)


@PluginBase.router.get("/txt2img/pano360")
async def txt2img_pano360_async(request: Txt2ImgPano360Request = Depends()):
    return await txt2img_pano360(request)
