import logging
from typing import Optional
from fastapi import BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
from PIL import Image
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.video_plugin import VideoPlugin
from utils.gpu_utils import set_seed
from utils.image_utils import get_image_from_request


class Img2VidRequest(BaseModel):
    image: str
    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 9.0
    seed: Optional[int] = -1
    fps: Optional[int] = 8
    interpolate_film: Optional[int] = 0
    interpolate_rife: Optional[int] = 1
    fast_interpolate: Optional[int] = False


class Img2VidGenXLPlugin(VideoPlugin):

    def __init__(self):
        super().__init__()
        self.name = "Image-to-video (GenXL)"
        self.description = "Image-to-video using i2vgenxl"

        from diffusers import I2VGenXLPipeline

        pipeline = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
        )

        pipeline.enable_model_cpu_offload(None, self.device)
        self.resources["pipeline"] = pipeline

    def generate(
        self,
        image: str | Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ):

        seed, generator = set_seed(seed, True)
        image = get_image_from_request(image) if isinstance(image, str) else image
        pipeline = self.resources["pipeline"]

        args = dict(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        if negative_prompt:
            args["negative_prompt"] = negative_prompt

        return pipeline(**args).frames[0]


@PluginBase.router.post("/img2vid/genxl")
async def img2vid(background_tasks: BackgroundTasks, req: Img2VidRequest):
    plugin: Img2VidGenXLPlugin = None
    try:
        plugin = await use_plugin(Img2VidGenXLPlugin)

        frames = plugin.generate(
            image=req.image,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
        )
        if not frames:
            return JSONResponse({"error": "Failed to generate video"})
        return plugin.video_response(
            background_tasks,
            frames,
            req,
        )
    except Exception as e:
        logging.error(e, exc_info=True)
    finally:
        if plugin is not None:
            release_plugin(plugin)


@PluginBase.router.get("/img2vid/genxl")
async def img2vid_from_url(
    background_tasks: BackgroundTasks, req: Img2VidRequest = Depends()
):
    return await img2vid(background_tasks, req)
