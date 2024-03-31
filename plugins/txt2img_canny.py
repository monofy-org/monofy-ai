import logging
from fastapi import Depends, HTTPException
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.stable_diffusion import StableDiffusionPlugin, Txt2ImgRequest
from utils.stable_diffusion_utils import filter_request
from settings import SD_USE_SDXL


class Txt2ImgCannyPlugin(StableDiffusionPlugin):

    name = "Stable Diffusion (Canny IP Adapter)"
    description = (
        "Stable Diffusion text-to-image using provided Canny outline as guidance."
    )
    instance = None

    def __init__(self, model=None):

        from diffusers import (
            StableDiffusionAdapterPipeline,
            StableDiffusionXLAdapterPipeline,
            T2IAdapter,
        )

        image_pipeline_type = (
            StableDiffusionXLAdapterPipeline
            if SD_USE_SDXL
            else StableDiffusionAdapterPipeline
        )

        if model is None:
            model = (
                "TencentARC/t2i-adapter-canny-sdxl-1.0"
                if SD_USE_SDXL
                else "TencentARC/t2i-adapter-canny-1.0"
            )

        logging.info(f"Loading model: {model}")

        adapter = T2IAdapter.from_pretrained(
            model,
            variant="fp16",
            use_safetensors=True,
            device_map="auto",
        )

        super().__init__(image_pipeline_type, adapter=adapter)

        self.resources["adapter"] = adapter        

    async def generate(
        self,
        req: Txt2ImgRequest,
    ):
        return await super().generate("txt2img", req)


@PluginBase.router.post("/txt2img/canny", tags=["Image Generation"])
async def txt2img(
    req: Txt2ImgRequest,
):
    plugin = None
    try:
        req = filter_request(req)
        plugin: Txt2ImgCannyPlugin = await use_plugin(Txt2ImgCannyPlugin)        
        result = await plugin.generate(req)
        return Txt2ImgCannyPlugin.format_response(req, result)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin is not None:
            release_plugin(Txt2ImgCannyPlugin)


@PluginBase.router.get("/txt2img/canny", tags=["Image Generation"])
async def txt2img_from_url(
    req: Txt2ImgRequest = Depends(),
):
    return await txt2img(req)
