import logging
import os
from fastapi import BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.video_plugin import VideoPlugin
from utils.file_utils import random_filename
from utils.gpu_utils import autodetect_device
from utils.video_utils import fix_video, get_fps, get_video_from_request


class Vid2DensePoseRequest(BaseModel):
    video: str


class Vid2DensePosePlugin(VideoPlugin):

    name = "DensePose"
    description = "Video to DensePose"
    instance = None

    def __init__(self):
        super().__init__()

    # Function to process video
    def generate(self, input_video_path):
        import torch
        import cv2
        import numpy as np
        from detectron2.engine import DefaultPredictor
        from densepose import add_densepose_config
        from densepose.vis.extractor import DensePoseResultExtractor
        from densepose.vis.densepose_results import (
            DensePoseResultsFineSegmentationVisualizer as Visualizer,
        )
        from detectron2.config import get_cfg

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(
            "submodules/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        )
        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        cfg.MODEL.DEVICE = autodetect_device(False)
        predictor = DefaultPredictor(cfg)

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # if height > width:
        #     logging.warning("Expanding dimensions of portrait mode frame")

        frames = []

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # if frame is portrait mode, EXPAND IT to make it a square. Do not rotate it.
            if height > width:
                diff = height - width
                padding = diff // 2
                frame = cv2.copyMakeBorder(
                    frame, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
                width = height

            with torch.no_grad():
                outputs = predictor(frame)["instances"]

            results = DensePoseResultExtractor()(outputs)
            cmap = cv2.COLORMAP_VIRIDIS
            # Visualizer outputs black for background, but we want the 0 value of
            # the colormap, so we initialize the array with that value
            arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
            out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)
            frames.append(out_frame)

        # Release resources
        cap.release()

        return frames


@PluginBase.router.post("/vid2densepose", tags=["Video Generation"])
async def vid2densepose(background_tasks: BackgroundTasks, req: Vid2DensePoseRequest):
    plugin: Vid2DensePosePlugin = None
    try:
        plugin = await use_plugin(Vid2DensePosePlugin)
        video_path = await get_video_from_request(req.video)
        frames = plugin.generate(video_path)
        return plugin.video_response(background_tasks, frames, fps=get_fps(video_path))

    except Exception as e:
        logging.error(f"Error in vid2densepose: {str(e)}", exc_info=True)
        return {"error": str(e)}
    finally:
        if plugin is not None:
            release_plugin(plugin)


@PluginBase.router.get("/vid2densepose", tags=["Video Generation"])
async def vid2densepose_from_url(req: Vid2DensePoseRequest = Depends()):
    return await vid2densepose(req)
