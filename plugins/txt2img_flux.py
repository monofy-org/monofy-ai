import torch
from fastapi import Depends
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from PIL import Image
from plugins.stable_diffusion import format_response
from settings import USE_XFORMERS
from utils.image_utils import image_to_base64_no_header


class Txt2ImgFluxRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512
    num_inference_steps: int = 4
    max_sequence_length: int = 256
    guidance_scale: float = 0
    return_json: bool = False


class Txt2ImgFluxPlugin(PluginBase):
    name = "Text-to-Image (Flux)"
    description = "txt2img_flux"
    instance = None

    def __init__(self):
        super().__init__()

        from diffusers import FluxPipeline

        from diffusers import (
            FluxTransformer2DModel,
        )

        # pipe: FluxPipeline = FluxPipeline(
        #     FlowMatchEulerDiscreteScheduler.from_pretrained("cocktailpeanut/xulf-s", subfolder="scheduler"),
        #     AutoencoderKL.from_pretrained("cocktailpeanut/xulf-s", subfolder="vae"),
        #     CLIPTextModel.from_pretrained("cocktailpeanut/xulf-s", subfolder="text_encoder").to(self.device),
        #     CLIPTokenizer.from_pretrained("cocktailpeanut/xulf-s", subfolder="tokenizer"),
        #     T5EncoderModel.from_pretrained("cocktailpeanut/xulf-s", subfolder="text_encoder_2").to(self.device),
        #     T5TokenizerFast.from_pretrained("cocktailpeanut/xulf-s", subfolder="tokenizer_2"),
        #     FluxTransformer2DModel.from_pretrained("cocktailpeanut/xulf-s", subfolder="transformer"),
        # ).to(self.device)

        transformer = FluxTransformer2DModel.from_single_file(
            "models/Flux/flux1_schnellFP8Kijai11GB.safetensors",
            device=self.device,
            torch_dtype=torch.bfloat16,
        )

        pipe: FluxPipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

        pipe.text_encoder.to(self.device)
        pipe.text_encoder_2.to(self.device)

        # if USE_XFORMERS:
        #     pipe.enable_xformers_memory_efficient_attention()

        pipe.enable_sequential_cpu_offload()

        self.resources["FluxPipeline"] = pipe

    async def generate(self, prompt: str, **kwargs):
        return self.resources["FluxPipeline"](prompt, **kwargs).images[0]


@PluginBase.router.post("/txt2img/flux", tags=["Text-to-Image"])
async def txt2img_flux(req: Txt2ImgFluxRequest):

    plugin: Txt2ImgFluxPlugin = None
    try:
        plugin = await use_plugin(Txt2ImgFluxPlugin)
        return_json = req.return_json
        req.__dict__.pop("return_json")
        image: Image.Image = await plugin.generate(**req.__dict__)
        image.save(".cache/test.png")
        return format_response(
            {"images": [image_to_base64_no_header(image)]} if return_json else image
        )

    finally:
        if plugin is not None:
            release_plugin(Txt2ImgFluxPlugin)


@PluginBase.router.get("/txt2img/flux", tags=["Text-to-Image"])
async def txt2img_flux_get(req: Txt2ImgFluxRequest = Depends()):
    return await txt2img_flux(req)
