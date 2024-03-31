import logging
from typing import Literal
from PIL import Image
from fastapi import Depends
from fastapi.responses import StreamingResponse
from modules.plugins import PluginBase, use_plugin, release_plugin
from classes.requests import Txt2ImgRequest
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS, SD_USE_SDXL
from utils.image_utils import (
    get_image_from_request,
    image_to_base64_no_header,
    image_to_bytes,
)
from utils.stable_diffusion_utils import (
    load_lora_settings,
    load_prompt_lora,
    postprocess,
)


default = Txt2ImgRequest()


class Txt2ImgControlNetPlugin(PluginBase):

    name = "Text-to-image (ControlNet)"
    description = "Text-to-image generation using ControlNet"
    instance = None

    def __init__(self):

        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            StableDiffusionXLControlNetPipeline,
        )
        from utils.gpu_utils import autodetect_dtype

        super().__init__()

        self.dtype = autodetect_dtype()

        model_path = SD_MODELS[SD_DEFAULT_MODEL_INDEX]

        controlnet_pipeline_type = (
            StableDiffusionXLControlNetPipeline
            if SD_USE_SDXL
            else StableDiffusionControlNetPipeline
        )

        canny_model = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
        ).to(dtype=self.dtype, device=self.device)

        txt2img_pipe = controlnet_pipeline_type.from_single_file(
            model_path,
            controlnet=canny_model,
            device=self.device,
            dtype=self.dtype,
        ).to(dtype=self.dtype, device=self.device)

        self.resources["txt2img"] = txt2img_pipe
        self.resources["canny_model"] = (canny_model,)
        self.resources["lora_settings"] = load_lora_settings()

    async def generate_image(
        self,
        req: Txt2ImgRequest,
        adapter: Literal[
            "canny"
        ] = "canny",  # TODO: Add support for other controlnet models
    ):
        pipe = self.resources["txt2img"]

        controlnet_image: Image.Image = get_image_from_request(req.image)

        # from scipy.signal import medfilt
        # controlnet_image = medfilt(controlnet_image, 3)

        # resize with anti-alias
        controlnet_image = controlnet_image.resize(
            (req.width, req.height), Image.LANCZOS
        )

        # smooth out the jagged outline
        from PIL import ImageFilter

        controlnet_image = controlnet_image.filter(ImageFilter.SMOOTH)

        lora_settings = self.resources["lora_settings"]
        load_prompt_lora(pipe, req, lora_settings)

        # generate image
        result = pipe(**req.__dict__, image=controlnet_image)

        image = result.images[0]

        image, json_response = await postprocess(self, image, req)

        pipe.unload_lora_weights()

        if req.return_json:
            json_response["images"] = [image_to_base64_no_header(image)]
            return json_response

        return StreamingResponse(
            image_to_bytes(image),
            media_type="image/png",
        )


@PluginBase.router.post(
    "/txt2img/controlnet", response_class=StreamingResponse, tags=["Image Generation"]
)
async def txt2img(
    req: Txt2ImgRequest,
    adapter: Literal[
        "canny"
    ] = "canny",  # TODO: Add support for other controlnet models
):
    logging.info(f"API txt2img Using controlnet: {adapter}")

    plugin = None

    try:
        plugin: Txt2ImgControlNetPlugin = await use_plugin(Txt2ImgControlNetPlugin)
        return await plugin.generate_image(req, adapter)

    except Exception as e:
        logging.error(f"Error loading plugin: {e}")
        raise e

    finally:
        if plugin is not None:
            release_plugin(Txt2ImgControlNetPlugin)


@PluginBase.router.get(
    "/txt2img/controlnet", response_class=StreamingResponse, tags=["Image Generation"]
)
async def txt2img_from_url(
    req: Txt2ImgRequest = Depends(),
    adapter: Literal[
        "canny"
    ] = "canny",  # TODO: Add support for other controlnet models
):
    plugin = None

    try:
        plugin: Txt2ImgControlNetPlugin = await use_plugin(Txt2ImgControlNetPlugin)
        return await plugin.generate_image(req, adapter)

    except Exception as e:
        logging.error(f"Error loading plugin: {e}")
        raise e

    finally:
        if plugin is not None:
            release_plugin(plugin)
