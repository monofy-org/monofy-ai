import logging

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from fastapi import BackgroundTasks
from PIL import Image

from modules.plugins import PluginBase, use_plugin
from plugins.video_plugin import Txt2VidRequest, VideoPlugin
from utils.console_logging import log_loading


class Txt2VidWan21Plugin(VideoPlugin):
    name = "Text-to-video (WAN 2.1)"
    description = "Text-to-video using WAN 2.1"
    instance = None

    def __init__(self):
        super().__init__()

    def load_model(self):
        if self.resources.get("pipe"):
            return self.resources.get("pipe")

        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        log_loading("Wan2.1", model_id)

        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")

        self.resources["pipe"] = pipe
        self.resources["vae"] = vae

        return pipe

    async def generate(self, req: Txt2VidRequest):
        pipe: WanPipeline = self.load_model()

        frames = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=480,
            width=832,
            num_frames=req.num_frames,
            guidance_scale=5.0,
        ).frames[0]

        # convery from ndarrays to Image.Image
        frames = [Image.fromarray(frame) for frame in frames]


        return frames


@PluginBase.router.post("/txt2vid/wan21")
async def txt2vid_wan21(background_tasks: BackgroundTasks, req: Txt2VidRequest):
    plugin: Txt2VidWan21Plugin = None
    try:
        plugin = await use_plugin(Txt2VidWan21Plugin)
        frames = await plugin.generate(req)
        return plugin.video_response(background_tasks, frames, req)

    except Exception as e:
        logging.error(e, exc_info=True)
