import logging

import torch
from diffusers.utils import export_to_video
from fastapi.responses import FileResponse

from classes.requests import Txt2VidRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.video_plugin import VideoPlugin
from utils.file_utils import random_filename
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
        pipe.enable_sequential_cpu_offload()

        self.resources["pipeline"] = pipe

    def generate(self, req: Txt2VidRequest):
        pipe = self.resources.get("pipeline")

        kwargs = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=704,
            height=480,
            num_frames=req.num_frames,  # 161,
            num_inference_steps=req.num_inference_steps,
        )

        if req.image:
            image = get_image_from_request(req.image)
            kwargs["image"] = image

        video = pipe(**kwargs).frames[0]
        output_path = random_filename("mp4")
        export_to_video(video, output_path, fps=24)
        return output_path

    def offload(self):
        from diffusers import LTXPipeline

        pipe: LTXPipeline = self.resources.get("pipeline")
        if pipe:
            pipe.maybe_free_model_hooks()


@PluginBase.router.post("/txt2vid/ltx")
async def txt2vid_ltx(request: Txt2VidRequest):
    plugin: Txt2VidLTXPlugin = None
    try:
        plugin = await use_plugin(Txt2VidLTXPlugin)
        output_path = plugin.generate(request)
        return FileResponse(output_path)
    except Exception as e:
        logging.error(e, exc_info=True)
        return {"error": str(e)}
    finally:
        if plugin:
            release_plugin(plugin)
