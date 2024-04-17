from fastapi import Depends
from classes.requests import Txt2ImgRequest
from modules.plugins import router, use_plugin, release_plugin
from plugins.stable_diffusion import StableDiffusionPlugin, format_response
from utils.image_utils import get_image_from_request
from utils.stable_diffusion_utils import postprocess

FACE_MODEL_INDEX = 1


@router.post("/txt2img/face", tags=["Image Generation"])
async def txt2img_face(req: Txt2ImgRequest):
    plugin = None
    try:
        plugin: StableDiffusionPlugin = await use_plugin(StableDiffusionPlugin)
        plugin._load_model(FACE_MODEL_INDEX)
        image = get_image_from_request(req.image)
        image, json_response = await postprocess(
            plugin,
            image,
            Txt2ImgRequest(face_prompt=req.prompt, negative_prompt=req.negative_prompt),
        )

        return format_response(req, json_response, image)

    except Exception as e:
        raise e
    finally:
        if plugin:
            release_plugin(StableDiffusionPlugin)


@router.get("/txt2img/face", tags=["Image Generation"])
async def txt2img_face_get(req: Txt2ImgRequest = Depends()):
    return await txt2img_face(req)
