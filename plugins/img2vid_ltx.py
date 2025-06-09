import logging

import torch
import tqdm
import tqdm.rich
from diffusers import LTXImageToVideoPipeline
from fastapi import BackgroundTasks, HTTPException

from classes.requests import Img2VidRequest, Txt2VidRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.video_plugin import VideoPlugin
from utils.image_utils import get_image_from_request
from utils.stable_diffusion_utils import set_seed


class Img2VidLTXPlugin(VideoPlugin):
    name = "Image-to-Video (LTX)"
    description = "Image-to-video using LTX"
    instance = None

    def __init__(self):
        super().__init__()

        pipe: LTXImageToVideoPipeline = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
        )        
        pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        self.resources["pipeline"] = pipe

    def generate(self, req: Txt2VidRequest):
        pipe: LTXImageToVideoPipeline = self.resources.get("pipeline")

        seed, generator = set_seed(req.seed, True)

        kwargs = dict(            
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=3.35,
            generator=generator,
        )

        if req.image:
            image = get_image_from_request(req.image, (req.width, req.height))
            kwargs["image"] = image

        original_progress_bar = pipe.progress_bar
        pipe.progress_bar = tqdm.tqdm

        result = pipe(**kwargs).frames[0]
        pipe.maybe_free_model_hooks()

        pipe.progress_bar = original_progress_bar

        return result

    def offload(self):
        from diffusers import LTXPipeline

        pipe: LTXPipeline = self.resources.get("pipeline")
        if pipe:
            pipe.maybe_free_model_hooks()


@PluginBase.router.post("/img2vid/ltx")
async def txt2vid_ltx(background_tasks: BackgroundTasks, request: Img2VidRequest):
    plugin: Img2VidLTXPlugin = None
    try:
        plugin = await use_plugin(Img2VidLTXPlugin)
        frames = plugin.generate(request)
        if frames:
            return plugin.video_response(background_tasks, frames, request)
        else:
            raise HTTPException(status_code=500, detail="Failed to generate video")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating video")
    finally:
        if plugin:
            release_plugin(plugin)
