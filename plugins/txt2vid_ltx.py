import logging

import torch
from fastapi import BackgroundTasks

from classes.requests import Txt2VidRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.video_plugin import VideoPlugin
from utils.gpu_utils import set_seed
from utils.image_utils import get_image_from_request


class Txt2VidLTXPlugin(VideoPlugin):
    name = "Text-to-Video (LTX)"
    description = "Text-to-video using LTX"
    instance = None

    def __init__(self):
        from diffusers import LTXPipeline

        super().__init__()

        pipe: LTXPipeline = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        self.resources["pipeline"] = pipe

    def generate(self, req: Txt2VidRequest):
        pipe = self.resources.get("pipeline")

        seed, generator = set_seed(req.seed or -1, True)

        kwargs = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,  # default is 704
            height=req.height,  # default is 480
            num_frames=req.num_frames,  # default is 161
            num_inference_steps=req.num_inference_steps,
            generator=generator,
        )

        if req.image:
            image = get_image_from_request(req.image)
            kwargs["image"] = image

        return pipe(**kwargs).frames[0]

    def offload(self):
        from diffusers import LTXPipeline

        pipe: LTXPipeline = self.resources.get("pipeline")
        if pipe:
            pipe.maybe_free_model_hooks()


@PluginBase.router.post("/txt2vid/ltx")
async def txt2vid_ltx(background_tasks: BackgroundTasks, request: Txt2VidRequest):
    plugin: Txt2VidLTXPlugin = None
    try:
        plugin = await use_plugin(Txt2VidLTXPlugin)
        frames = plugin.generate(request)
        return plugin.video_response(background_tasks, frames, request)

    except Exception as e:
        logging.error(e, exc_info=True)
        return {"error": str(e)}
    finally:
        if plugin:
            release_plugin(plugin)
