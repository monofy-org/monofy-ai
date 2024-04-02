import gc
import io
import logging
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.gpu_utils import set_seed
from utils.image_utils import image_to_base64_no_header
from settings import USE_XFORMERS


class Txt2ImgCascadePlugin(PluginBase):

    name = "Stable Cascade"
    description = "Cascade text-to-image generation"
    instance = None

    def __init__(self):
        import torch

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        negative_prompt: str = "",
        guidance_scale: float = 4.0,
        num_inference_steps: int = 10,
        num_inference_steps_prior: int = 20,
        seed: int = -1,
    ):
        import torch
        from diffusers import (
            DiffusionPipeline,
            StableCascadePriorPipeline,
            StableCascadeDecoderPipeline,
        )

        prior: DiffusionPipeline = StableCascadePriorPipeline.from_pretrained(
            "stabilityai/stable-cascade-prior",
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        if USE_XFORMERS and torch.cuda.is_available():
            prior.enable_xformers_memory_efficient_attention()

        set_seed(seed)

        prior_output = prior(
            prompt=prompt,
            height=width,
            width=height,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps_prior,
        )

        print("Finished step 1")

        prior.maybe_free_model_hooks()
        del prior
        # gc.collect()
        # if torch.cuda.is_available():
        #    torch.cuda.empty_cache()

        decoder: DiffusionPipeline = StableCascadeDecoderPipeline.from_pretrained(
            "stabilityai/stable-cascade",
            torch_dtype=torch.float16,
        ).to(self.device)

        if USE_XFORMERS and torch.cuda.is_available():
            decoder.enable_xformers_memory_efficient_attention()

        image = decoder(
            image_embeddings=prior_output.image_embeddings.half(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=num_inference_steps,
        ).images[0]

        decoder.maybe_free_model_hooks()
        del decoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image


@PluginBase.router.post("/txt2img/cascade", tags=["Image Generation"])
async def txt2img_cascade(
    req: Txt2ImgRequest,
):
    try:
        plugin: Txt2ImgCascadePlugin = await use_plugin(Txt2ImgCascadePlugin)

        num_inference_steps = req.num_inference_steps or 10

        image = plugin.generate_image(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            negative_prompt=req.negative_prompt,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            num_inference_steps_prior=num_inference_steps * 2,
            seed=req.seed,
        )

        if req.return_json:

            return {
                "images": [image_to_base64_no_header(image)],
                "objects": [],
                "detections": [],
            }

        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        return StreamingResponse(image_bytes, media_type="image/png")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        release_plugin(Txt2ImgCascadePlugin)


@PluginBase.router.get("/txt2img/cascade", tags=["Image Generation"])
async def txt2img_cascade_from_url(
    req: Txt2ImgRequest = Depends(),
):
    return await txt2img_cascade(req)
