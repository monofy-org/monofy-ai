import logging
from fastapi import BackgroundTasks, Depends
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from pydantic import BaseModel
from typing import Optional
from plugins.img_depth_anything import DepthAnythingPlugin
from plugins.txt2img_depth import Txt2ImgDepthMidasPlugin
from plugins.extras.youtube import create_grid, download_youtube_video
from plugins.video_plugin import VideoPlugin
from utils.file_utils import download_to_cache
from utils.gpu_utils import clear_gpu_cache
from utils.image_utils import image_to_base64_no_header


class Vid2VidRequest(BaseModel):
    video: str
    prompt: str
    negative_prompt: Optional[str] = ""
    seed: Optional[int] = -1
    rows: Optional[int] = 2
    cols: Optional[int] = 2


class Vid2VidPlugin(VideoPlugin):

    name = "Vid2Vid (frame interpolation)"
    description = "Vid2Vid using grid combining/slicing and frame interpolation"
    instance = None
    plugins = [DepthAnythingPlugin, Txt2ImgDepthMidasPlugin, VideoPlugin]

    def __init__(self):
        super().__init__()        

    async def vid2vid(self, request: Vid2VidRequest):
        video_path = await get_video_from_request(request.video)
        print(video_path)
        grid = create_grid(video_path, request.rows, request.cols)
        size = grid.size
        depth: DepthAnythingPlugin = await use_plugin(DepthAnythingPlugin, True)
        grid = await depth.generate_depthmap(grid)
        grid = grid.resize(size)

        txt2img: Txt2ImgDepthMidasPlugin = await use_plugin(Txt2ImgDepthMidasPlugin, True)
        image = await txt2img.generate(
            Txt2ImgRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or "",
                seed=request.seed,
                image=image_to_base64_no_header(grid),
                width=grid.width,
                height=grid.height,
            )
        )      

        frames = []
        # split up image into separate images again
        w = image.width // request.cols
        h = image.height // request.rows
        for i in range(request.rows):
            for j in range(request.cols):
                frame = image.crop((j * w, i * h, (j + 1) * w, (i + 1) * h))
                frames.append(frame)

        return frames


async def get_video_from_request(video: str) -> str:
    if "youtube.com" in video:
        return await download_youtube_video(video)
    else:
        return download_to_cache(video)


@PluginBase.router.post("/vid2vid", tags=["Video Generation"])
async def vid2vid(background_tasks: BackgroundTasks, request: Vid2VidRequest):
    plugin = None
    try:
        plugin: Vid2VidPlugin = await use_plugin(Vid2VidPlugin, True)
        images = await plugin.vid2vid(request)
        clear_gpu_cache()
        return plugin.video_response(background_tasks, plugin, images, fps=8)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin is not None:
            await release_plugin(Vid2VidPlugin)


@PluginBase.router.get("/vid2vid", tags=["Video Generation"])
async def vid2vid_from_url(background_tasks: BackgroundTasks, request: Vid2VidRequest = Depends()):
    return await vid2vid(background_tasks, request)
