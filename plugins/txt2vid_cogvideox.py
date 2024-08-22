from fastapi import BackgroundTasks, Depends
import torch
from typing import Optional
from pydantic import BaseModel

from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.video_plugin import VideoPlugin


class Txt2VidCogVideoXRequest(BaseModel):
    prompt: str
    guidance_scale: Optional[float] = 6.0
    num_inference_steps: Optional[int] = 50


class Txt2VidCogVideoXPlugin(VideoPlugin):
    name = "CogVideoX"
    description = "Text-to-video using CogVideoX"
    instance = None

    def __init__(self):
        from diffusers import CogVideoXPipeline

        super().__init__()

        self.dtype = torch.float16

        pipe: CogVideoXPipeline = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b", torch_dtype=self.dtype
        ).to(self.device, dtype=self.dtype)

        pipe.enable_sequential_cpu_offload()

        self.resources["pipeline"] = pipe

    def generate(
        self,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
    ):
        from diffusers import CogVideoXPipeline

        torch.cuda.empty_cache()

        pipe: CogVideoXPipeline = self.resources["pipeline"]

        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=None,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=226,
            device=self.device,
            dtype=self.dtype,
        )

        video = pipe(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=torch.zeros_like(prompt_embeds),
        ).frames[0]

        return video


@PluginBase.router.post("/txt2vid/cogvideox", tags=["Video Generation (text-to-video)"])
async def txt2vid_cogvideox(
    background_tasks: BackgroundTasks, req: Txt2VidCogVideoXRequest
):
    plugin: Txt2VidCogVideoXPlugin = None
    try:
        plugin = await use_plugin(Txt2VidCogVideoXPlugin)
        video = plugin.generate(
            prompt=req.prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
        )
        return plugin.video_response(background_tasks, video)
    except Exception as e:
        raise e
    finally:
        if plugin:
            release_plugin(Txt2VidCogVideoXPlugin)


@PluginBase.router.get("/txt2vid/cogvideox", tags=["Video Generation (text-to-video)"])
async def txt2vid_cogvideox_stream(
    background_tasks: BackgroundTasks, req: Txt2VidCogVideoXRequest = Depends()
):
    return await txt2vid_cogvideox(background_tasks, req)
