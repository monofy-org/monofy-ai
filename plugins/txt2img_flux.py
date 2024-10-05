import os
import torch
from fastapi import Depends
from classes.requests import Txt2ImgRequest
from modules.filter import filter_request
from modules.plugins import PluginBase, release_plugin, use_plugin
from PIL import Image
from plugins.stable_diffusion import format_response
from utils.console_logging import log_loading
from utils.gpu_utils import (
    autodetect_device,
    clear_gpu_cache,
    disable_hot_loading,
    enable_hot_loading,
    set_seed,
)
from utils.image_utils import image_to_base64_no_header
from utils.stable_diffusion_utils import load_lora_settings, load_prompt_lora


class Txt2ImgFluxPlugin(PluginBase):
    name = "Text-to-image (Flux)"
    description = "Text-to-image using Flux"
    instance = None
    device = autodetect_device(allow_accelerate=False)

    def __init__(self):
        super().__init__()

        from diffusers import FluxPipeline, FluxTransformer2DModel

        # from classes.flux4bit import T5EncoderModel, FluxTransformer2DModel

        clear_gpu_cache()

        model = "flux1_schnellFP8Kijai11GB.safetensors"

        log_loading("Flux", model)

        # TODO: figure out 4-bit model loading

        # from transformers import BitsAndBytesConfig

        # nf4_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.float16,
        # )

        # text_encoder_2: T5EncoderModel = T5EncoderModel.from_pretrained(
        #     "HighCWu/FLUX.1-dev-4bit",
        #     subfolder="text_encoder_2",
        #     torch_dtype=torch.bfloat16,
        #     # hqq_4bit_compute_dtype=torch.float32,
        # )

        # transformer: FluxTransformer2DModel = FluxTransformer2DModel.from_pretrained(
        #     "HighCWu/FLUX.1-dev-4bit",
        #     subfolder="transformer",
        #     torch_dtype=torch.bfloat16,
        # )

        # pipe: FluxPipeline = FluxPipeline.from_pretrained(
        #     "black-forest-labs/FLUX.1-dev",
        #     text_encoder_2=text_encoder_2,
        #     transformer=transformer,
        #     torch_dtype=torch.bfloat16,
        # )

        disable_hot_loading()

        base_model = "black-forest-labs/FLUX.1-" + (
            "schnell"
            if ("schnell" in model.lower() or "hybrid" in model.lower())
            else "dev"
        )

        transformer = FluxTransformer2DModel.from_single_file(
            os.path.join("models", "Flux", model),
            torch_dtype=torch.float16,
        )

        pipe: FluxPipeline = FluxPipeline.from_pretrained(
            base_model,
            transformer=transformer,
            torch_dtype=torch.float16,
            # quantization_config=nf4_config,            
        )

        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()

        enable_hot_loading()

        self.resources["FluxPipeline"] = pipe
        self.last_loras = []

    async def generate(self, prompt: str, **kwargs):

        _, generator = set_seed(kwargs.get("seed") or -1, True)

        pipe = self.resources["FluxPipeline"]

        lora_settings = load_lora_settings("flux")

        loras = load_prompt_lora(
            pipe,
            Txt2ImgRequest(prompt=prompt, **kwargs),
            lora_settings,
            self.last_loras,
        )
        self.last_loras = loras

        kwargs["generator"] = generator
        kwargs.pop("seed")

        # Unused/unsupported by Flux
        kwargs.pop("model_index")
        kwargs.pop("negative_prompt")
        kwargs.pop("scheduler")
        kwargs.pop("face_prompt")
        kwargs.pop("upscale")
        kwargs.pop("strength")
        kwargs.pop("freeu")
        kwargs.pop("hi")
        kwargs.pop("hyper")
        kwargs.pop("invert")
        kwargs.pop("tiling")
        kwargs.pop("controlnet")
        kwargs.pop("use_refiner")
        kwargs.pop("image2")

        # TODO: add support for these
        nsfw = kwargs.pop("nsfw")
        auto_lora = kwargs.pop("auto_lora")
        adapter = kwargs.pop("adapter")
        image = kwargs.pop("image")
        lora_strength = kwargs.pop("lora_strength")

        return pipe(prompt, **kwargs).images[0]


@PluginBase.router.post("/txt2img/flux", tags=["Image Generation"])
async def txt2img_flux(req: Txt2ImgRequest):

    if not req.num_inference_steps:
        req.num_inference_steps = 4

    req = filter_request(req)

    plugin: Txt2ImgFluxPlugin = None
    try:
        plugin = await use_plugin(Txt2ImgFluxPlugin)
        return_json = req.return_json
        req.__dict__.pop("return_json")
        image: Image.Image = await plugin.generate(**req.__dict__)
        return format_response(
            {"images": [image_to_base64_no_header(image)]} if return_json else image
        )

    finally:
        if plugin is not None:
            release_plugin(Txt2ImgFluxPlugin)


@PluginBase.router.get("/txt2img/flux", tags=["Image Generation"])
async def txt2img_flux_get(req: Txt2ImgRequest = Depends()):
    return await txt2img_flux(req)
