import logging
import torch
from fastapi import Depends
from fastapi.responses import StreamingResponse
from modules.plugins import PluginBase, use_plugin, release_plugin
from modules.filter import filter_request
from plugins.stable_diffusion import format_response
from classes.requests import Txt2ImgRequest
from utils.image_utils import get_image_from_request
from PIL import ImageFilter
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS
from utils.stable_diffusion_utils import (
    load_lora_settings,
    load_prompt_lora,
    postprocess,
)

controlnets_xl = {
    "inpaint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    # "inpaint": "destitech/controlnet-inpaint-dreamer-sdxl",
    "canny": "diffusers/controlnet-canny-sdxl-1.0-small",
    "depth": "diffusers/controlnet-depth-sdxl-1.0",
    "openpose": "OzzyGT/controlnet-openpose-sdxl-1.0",
    "tile": "TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic",
    "qr": "monster-labs/control_v1p_sd15_qrcode_monster",
}


class Txt2ImgControlNetPlugin(PluginBase):

    name = "Text-to-image (ControlNet)"
    description = "Text-to-image generation using ControlNet"
    instance = None

    def __init__(self):

        from utils.gpu_utils import autodetect_dtype

        super().__init__()

        self.dtype = autodetect_dtype()
        self.last_loras = None

        self.current_controlnet: str = None
        self.current_model_index: int = None

        self.resources["lora_settings"] = load_lora_settings()
        self.resources["controolnet_model"] = None

    def _load_model(self, model_index: int, controlnet: str):

        if controlnet not in controlnets_xl:
            raise ValueError("ControlNet model not found")

        if (
            controlnet == self.current_controlnet
            and model_index == self.current_model_index
        ):
            return

        model_path: str = SD_MODELS[model_index]

        use_sdxl = "xl" in model_path.lower()

        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            StableDiffusionXLControlNetPipeline,
        )

        controlnet_pipeline_type = (
            StableDiffusionXLControlNetPipeline
            if use_sdxl
            else StableDiffusionControlNetPipeline
        )

        if controlnet != self.current_controlnet:
            controlnet_model = ControlNetModel.from_pretrained(
                # diffusers/controlnet-canny-sdxl-1.0
                controlnets_xl[controlnet],
                torch_dtype=torch.float16,
            ).to(dtype=self.dtype, device=self.device)
            self.resources["controlnet_model"] = controlnet_model
            self.resources["txt2img"].controlnet = controlnet_model
            self.current_controlnet = controlnet            

        if model_index != self.current_model_index:
            txt2img_pipe = controlnet_pipeline_type.from_single_file(
                model_path,
                controlnet=self.resources["controlnet_model"],
                device=self.device,
                dtype=self.dtype,
            ).to(dtype=self.dtype, device=self.device)

        self.resources["txt2img"] = txt2img_pipe
        self.resources["controlnet_model"] = controlnet_model

    async def generate_image(
        self,
        req: Txt2ImgRequest,
    ):
        if not req.controlnet:
            raise ValueError("ControlNet model not specified")

        self._load_model(req.model_index or SD_DEFAULT_MODEL_INDEX, req.controlnet)
        pipe = self.resources["txt2img"]

        req = filter_request(req)

        outline_image = get_image_from_request(req.image, (req.width, req.height))

        # from scipy.signal import medfilt
        # controlnet_image = medfilt(controlnet_image, 3)

        # smooth out the jagged outline

        outline_image = outline_image.filter(ImageFilter.SMOOTH)

        lora_settings = self.resources["lora_settings"]
        self.last_loras = load_prompt_lora(pipe, req, lora_settings, self.last_loras)

        # generate image
        outline_image = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps or SD_DEFAULT_STEPS,
            guidance_scale=req.guidance_scale,
            image=outline_image,
        ).image

        outline_image, json_response = await postprocess(self, outline_image, req)

        pipe.unload_lora_weights()

        return format_response(req, json_response, outline_image)


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
