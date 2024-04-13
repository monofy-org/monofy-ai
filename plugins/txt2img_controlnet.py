import logging
from typing import Literal
from fastapi import Depends
from fastapi.responses import StreamingResponse
from modules.plugins import PluginBase, use_plugin, release_plugin
from modules.filter import filter_request
from plugins.stable_diffusion import format_response
from classes.requests import Txt2ImgRequest
from utils.image_utils import get_image_from_request
from settings import SD_DEFAULT_MODEL_INDEX, SD_DEFAULT_STEPS, SD_MODELS, SD_USE_SDXL
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
        self.last_loras = None

        model_path = SD_MODELS[SD_DEFAULT_MODEL_INDEX]

        controlnet_pipeline_type = (
            StableDiffusionXLControlNetPipeline
            if SD_USE_SDXL
            else StableDiffusionControlNetPipeline
        )

        canny_model = ControlNetModel.from_pretrained(
            # diffusers/controlnet-canny-sdxl-1.0
            "monster-labs/control_v1p_sd15_qrcode_monster",
            torch_dtype=torch.float16,
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
    ):
        pipe = self.resources["txt2img"]

        image = get_image_from_request(req.image, (req.width, req.height))

        req = filter_request(req)

        # from scipy.signal import medfilt
        # controlnet_image = medfilt(controlnet_image, 3)

        # smooth out the jagged outline
        from PIL import ImageFilter

        image = image.filter(ImageFilter.SMOOTH)

        lora_settings = self.resources["lora_settings"]
        self.last_loras = load_prompt_lora(pipe, req, lora_settings, self.last_loras)

        # generate image
        image = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps or SD_DEFAULT_STEPS,
            guidance_scale=req.guidance_scale,
            image=image,
        ).image

        image, json_response = await postprocess(self, image, req)

        pipe.unload_lora_weights()

        return format_response(req, json_response, image)


@PluginBase.router.post(
    "/txt2img/controlnet", response_class=StreamingResponse, tags=["Image Generation"]
)
async def txt2img(
    req: Txt2ImgRequest,
):
    plugin: Txt2ImgControlNetPlugin = None

    try:
        plugin = await use_plugin(Txt2ImgControlNetPlugin)
        return await plugin.generate_image(req)

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
):
    return await txt2img(req)
