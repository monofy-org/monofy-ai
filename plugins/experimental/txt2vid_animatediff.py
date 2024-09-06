import logging
from fastapi import BackgroundTasks, Depends, HTTPException
import modules.plugins as plugins
from classes.requests import Txt2VidRequest
from plugins.video_plugin import VideoPlugin
from submodules.AnimateDiff.app import AnimateController


class Txt2VidAnimateDiffPlugin(VideoPlugin):
    name = "Text-to-Video (AnimateDiff)"
    description = "Text-to-video generation using AnimateDiff"
    instance = None
    plugins = ["StableDiffusionPlugin"]

    def __init__(self):
        super().__init__()

        animator = AnimateController()
        self.resources["animator"] = animator
        self.resources["pipeline"] = animator.pipeline

    async def generate(
        self,
        req: Txt2VidRequest,
    ):
        pipe = self.resources["pipeline"]

        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            sampler=req.sampler,
            sample_step=req.sample_step,
            width=req.width,
            length=req.length,
            height=req.height,
            cfg_scale=req.cfg_scale,
            seed=req.seed,
        ).videos[0]

        print(result)

        return result


@plugins.router.post("/txt2vid/animatediff", tags=["Text-to-Video"])
async def txt2vid_animatediff(background_tasks: BackgroundTasks, req: Txt2VidRequest):
    plugin: Txt2VidAnimateDiffPlugin = None
    try:
        plugin = await plugins.use_plugin(Txt2VidAnimateDiffPlugin)
        frames = await plugin.generate(**req.__dict__)

        return plugin.video_response(
            background_tasks,
            frames,
            req.fps,
            req.interpolate_film,
            req.interpolate_rife,
            req.fast_interpolate,
        )
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
    finally:
        if plugin is not None:
            plugins.release_plugin(plugin)


@plugins.router.get("/txt2vid/animatediff", tags=["Text-to-Video"])
async def txt2vid_animatediff_from_url(
    background_tasks: BackgroundTasks, req: Txt2VidRequest = Depends()
):
    return await txt2vid_animatediff(background_tasks, req)
