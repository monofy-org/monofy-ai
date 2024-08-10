from fastapi import Depends
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from PIL import Image
from plugins.stable_diffusion import format_response


class Txt2ImgFluxRequest(BaseModel):
    prompt: str    
    width: int = 512
    height: int = 512
    num_inference_steps: int = 4
    max_sequence_length: int = 256
    guidance_scale: float = 0


class Txt2ImgFluxPlugin(PluginBase):
    name = "Text-to-Image (Flux)"
    description = "txt2img_flux"
    instance = None

    def __init__(self):
        super().__init__()

        import torch
        from diffusers import FluxPipeline

        pipe: FluxPipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()

        self.resources["FluxPipeline"] = pipe

    def generate(self, prompt, **kwargs):
        from diffusers import FluxPipeline

        pipe: FluxPipeline = self.resources["FluxPipeline"]
        return pipe(prompt, **kwargs).images[0]


@PluginBase.router.post("/txt2img/flux", tags=["Text-to-Image"])
async def txt2img_flux(request: Txt2ImgFluxRequest):

    plugin: Txt2ImgFluxPlugin = None
    try:
        plugin = await use_plugin(Txt2ImgFluxPlugin)
        image: Image.Image = plugin.generate(**request.__dict__)
        return format_response(image)

    finally:
        if plugin is not None:
            release_plugin(Txt2ImgFluxPlugin)


@PluginBase.router.get("/txt2img/flux", tags=["Text-to-Image"])
async def txt2img_flux_get(request: Txt2ImgFluxRequest = Depends()):
    return await txt2img_flux(request)
