import logging
from fastapi import Depends, HTTPException
import torch
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.stable_diffusion import (
    StableDiffusionPlugin,
    Txt2ImgRequest,
    format_response,
)
from modules.filter import filter_request


class Txt2ImgCannyPlugin(StableDiffusionPlugin):

    name = "Stable Diffusion (Canny IP Adapter)"
    description = (
        "Stable Diffusion text-to-image using provided Canny outline as guidance."
    )
    instance = None

    def __init__(self, adapter_repo_or_path=None):

        from diffusers import (
            # StableDiffusionAdapterPipeline,
            StableDiffusionXLAdapterPipeline,
            T2IAdapter,
        )

        image_pipeline_type = (
            StableDiffusionXLAdapterPipeline
            # if SD_USE_SDXL
            # else StableDiffusionAdapterPipeline
        )

        if adapter_repo_or_path is None:
            adapter_repo_or_path = (
                "TencentARC/t2i-adapter-canny-sdxl-1.0"
                #if SD_USE_SDXL
                #else "TencentARC/t2i-adapter-canny-1.0"
            )

        logging.info(f"Loading model: {adapter_repo_or_path}")

        adapter: T2IAdapter = T2IAdapter.from_pretrained(
            adapter_repo_or_path,
            variant="fp16",
            use_safetensors=True,
        )

        if torch.cuda.is_available():
            adapter.cuda()

        super().__init__(image_pipeline_type, adapter=adapter)

        self.resources["adapter"] = adapter

    async def generate(
        self,
        req: Txt2ImgRequest,
    ):
        return await super().generate("txt2img", req, adapter_conditioning_scale=0.8, adapter_conditioning_factor=1)


@PluginBase.router.post("/txt2img/canny", tags=["Image Generation"])
async def txt2img(
    req: Txt2ImgRequest,
):
    plugin = None
    try:
        req = filter_request(req)        
        plugin: Txt2ImgCannyPlugin = await use_plugin(Txt2ImgCannyPlugin)
        plugin.load_model(req.model_index)
        plugin.resources.get("pipeline").set_ip_adapter_scale(0.3)
        response = await plugin.generate(req)
        return format_response(req, response)
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
