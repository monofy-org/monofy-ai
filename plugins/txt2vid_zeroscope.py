import logging
from fastapi import BackgroundTasks, Depends
from classes.requests import Txt2VidRequest
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.gpu_utils import clear_gpu_cache
from utils.video_utils import video_response
from settings import TXT2VID_MAX_FRAMES


class Txt2VidZeroscopePlugin(PluginBase):

    name = "txt2vid_zeroscope"
    description = "Zeroscope text-to-video generation"
    instance = None

    def __init__(self):
        from utils.gpu_utils import autodetect_dtype
        from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

        super().__init__()

        self.dtype = autodetect_dtype(bf16_allowed=False)

        pipe = DiffusionPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w",
        ).to(self.device, dtype=self.dtype)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_vae_slicing()

        self.resources["DiffusionPipeline"] = pipe

    async def generate(self, req: Txt2VidRequest):
        pipe = self.resources["DiffusionPipeline"]

        frames = pipe(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            num_frames=req.num_frames,
        ).frames[0]

        return frames


@PluginBase.router.post("/txt2vid/zeroscope", tags=["Video Generation (text-to-video)"])
async def txt2vid(
    background_tasks: BackgroundTasks,
    req: Txt2VidRequest,
):
    plugin = None
    try:
        plugin = await use_plugin(Txt2VidZeroscopePlugin)
        req.num_frames = min(req.num_frames, TXT2VID_MAX_FRAMES)

        frames = await plugin.generate(req)

        clear_gpu_cache()

        return video_response(
            background_tasks, frames, req.interpolate, req.fast_interpolate, req.fps
        )
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin:
            release_plugin(Txt2VidZeroscopePlugin)


@PluginBase.router.get("/txt2vid/zeroscope", tags=["Video Generation (text-to-video)"])
async def txt2vid_get(
    background_tasks: BackgroundTasks,
    req: Txt2VidRequest = Depends(),
):
    return await txt2vid(background_tasks, req)
