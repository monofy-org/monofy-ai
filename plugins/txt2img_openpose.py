import logging
from fastapi import Depends, HTTPException
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from modules.filter import filter_request
from plugins.stable_diffusion import format_response
from plugins.txt2img_canny import Txt2ImgCannyPlugin


class Txt2ImgOpenPosePlugin(Txt2ImgCannyPlugin):
    name = "Stable Diffusion (OpenPose IP Adapter)"
    description = "Stable Diffusion text-to-image using provided depth map as guidance."
    instance = None

    def __init__(self):
        super().__init__(
            "TencentARC/t2i-adapter-openpose-sdxl-1.0"
            # if SD_USE_SDXL
            # else "TencentARC/t2i-adapter-depth-midas-1.0"
            ,
            None
        )


@PluginBase.router.post("/txt2img/openpose", tags=["Image Generation"])
async def txt2img(
    req: Txt2ImgRequest,
):
    if not req.image:
        raise HTTPException(status_code=400, detail="No image provided")

    plugin: Txt2ImgOpenPosePlugin = None
    try:
        req = filter_request(req)
        req.scheduler = req.scheduler or "sde"
        plugin = await use_plugin(Txt2ImgOpenPosePlugin)
        response = await plugin.generate(req)
        return format_response(response)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin is not None:
            release_plugin(Txt2ImgOpenPosePlugin)


@PluginBase.router.get("/txt2img/openpose", tags=["Image Generation"])
async def txt2img_from_url(
    req: Txt2ImgRequest = Depends(),
):
    return await txt2img(req)
