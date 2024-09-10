import logging
import os
from fastapi import Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.video_plugin import VideoPlugin
from settings import CACHE_PATH
from utils.console_logging import log_recycle
from utils.file_utils import (
    cached_snapshot,
    download_to_cache,
    random_filename,
    url_hash,
)
from utils.image_utils import get_image_from_request
from utils.video_utils import remove_audio


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


class Img2VidLivePortraitRequest(BaseModel):
    image: str
    video: str
    relative_motion: Optional[bool] = True
    do_crop: Optional[bool] = True
    paste_back: Optional[bool] = True
    include_audio: Optional[bool] = True
    mirror: Optional[bool] = False


class Img2VidLivePortraitPlugin(VideoPlugin):

    name = "LivePortrait"
    description = "Image to video generation using LivePortrait"
    instance = None

    def __init__(self):
        import tyro
        from submodules.LivePortrait.src.config.argument_config import ArgumentConfig
        from submodules.LivePortrait.src.config.inference_config import InferenceConfig
        from submodules.LivePortrait.src.config.crop_config import CropConfig

        super().__init__()

        from classes.liveportrait_pipeline import BasicPipeline

        args = tyro.cli(ArgumentConfig)

        # specify configs for inference
        inference_cfg = partial_fields(
            InferenceConfig, args.__dict__
        )  # use attribute of args to initial InferenceConfig
        crop_cfg = partial_fields(
            CropConfig, args.__dict__
        )  # use attribute of args to initial CropConfig

        models_path = cached_snapshot("KwaiVGI/LivePortrait")

        crop_cfg.insightface_root = os.path.join(models_path, "insightface")

        crop_cfg.landmark_ckpt_path = os.path.join(
            models_path, "liveportrait", "landmark.onnx"
        )

        inference_cfg.checkpoint_F = os.path.join(
            models_path,
            "liveportrait/base_models",
            "appearance_feature_extractor.pth",
        )

        inference_cfg.checkpoint_M = os.path.join(
            models_path,
            "liveportrait/base_models",
            "motion_extractor.pth",
        )

        inference_cfg.checkpoint_G = os.path.join(
            models_path,
            "liveportrait/base_models",
            "spade_generator.pth",
        )

        inference_cfg.checkpoint_W = os.path.join(
            models_path,
            "liveportrait/base_models",
            "warping_module.pth",
        )

        inference_cfg.checkpoint_S = os.path.join(
            models_path,
            "liveportrait/retargeting_models",
            "stitching_retargeting_module.pth",
        )

        pipeline = BasicPipeline(
            inference_cfg=inference_cfg, crop_cfg=crop_cfg, args=args
        )

        self.resources["pipeline"] = pipeline

    async def generate(
        self,
        req: Img2VidLivePortraitRequest,
    ):

        from classes.liveportrait_pipeline import BasicPipeline

        pipeline: BasicPipeline = self.resources["pipeline"]

        hash = url_hash(req.video)
        pkl = f"{CACHE_PATH}/{hash}.pkl"
        if os.path.exists(pkl):
            log_recycle("Reusing motion template: " + pkl)
            pipeline.args.driving_info = pkl
        else:
            pkl = None

        image_path = get_image_from_request(
            req.image, mirror=req.mirror, return_path=True
        )
        video_path = pkl or download_to_cache(req.video, "mp4")

        if not os.path.exists(image_path):
            raise Exception("Failed to read image: " + image_path)
        if not os.path.exists(video_path):
            raise Exception("Failed to read video: " + video_path)

        def gpu_wrapped_execute_video(*args, **kwargs):
            return pipeline.execute_video(*args, **kwargs)

        def gpu_wrapped_execute_image(*args, **kwargs):
            return pipeline.execute_image(*args, **kwargs)
        
        try:
            path, filename = gpu_wrapped_execute_video(
                input_image_path=image_path,
                input_video_path=video_path,
                flag_relative_input=req.relative_motion,
                flag_do_crop_input=req.do_crop,
                flag_remap_input=req.paste_back,
                flag_crop_driving_video_input=True,
            )
        except Exception:            
            raise Exception("Failed to generate video")

        if not (filename and os.path.exists(path)):
            raise Exception("Failed to generate video")

        if req.include_audio:
            if pkl:

                original_video = f"{CACHE_PATH}/{hash}.mp4"

                if not os.path.exists(original_video):
                    logging.warning("Original video not found, audio will be removed")
                else:
                    from ffmpy import FFmpeg

                    temp_file = random_filename("mp4")

                    output = FFmpeg(
                        inputs={
                            original_video: None,
                            path: None,
                        },
                        outputs={temp_file: "-c:v copy -c:a aac -strict experimental"},
                    )

                    output.run()
                    os.remove(path)
                    os.rename(temp_file, path)

            else:
                if not req.include_audio:
                    path = remove_audio(path, delete_old_file=True)

        return path, os.path.basename(path)


@PluginBase.router.post(
    "/img2vid/liveportrait", tags=["Video Generation (image-to-video)"]
)
async def img2vid_liveportrait(req: Img2VidLivePortraitRequest):
    plugin: Img2VidLivePortraitPlugin = None
    filename = None
    try:
        plugin: Img2VidLivePortraitPlugin = await use_plugin(Img2VidLivePortraitPlugin)
        full_path, filename = await plugin.generate(req)

        return FileResponse(full_path, filename=filename)
    except Exception as e:
        logging.error(e, exc_info=True)
        return {"error": str(e)}
    finally:
        if plugin:
            release_plugin(plugin)
        # if filename:
        #     os.remove(full_path)


@PluginBase.router.get(
    "/img2vid/liveportrait", tags=["Video Generation (image-to-video)"]
)
async def img2vid_liveportrait_get(req: Img2VidLivePortraitRequest = Depends()):
    return await img2vid_liveportrait(req)
