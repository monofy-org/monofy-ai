import logging
import os
from fastapi import Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.file_utils import random_filename
from utils.video_utils import fix_video


class Vid2DensePoseRequest(BaseModel):
    video: str


class Vid2DensePosePlugin(PluginBase):

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

        output_video_path = random_filename("mp4")

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(
            "submodules/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        )
        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        cfg.MODEL.DEVICE = self.device
        predictor = DefaultPredictor(cfg)

        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            with torch.no_grad():
                outputs = predictor(frame)["instances"]

            results = DensePoseResultExtractor()(outputs)
            cmap = cv2.COLORMAP_VIRIDIS
            # Visualizer outputs black for background, but we want the 0 value of
            # the colormap, so we initialize the array with that value
            arr = cv2.applyColorMap(np.zeros((height, width), dtype=np.uint8), cmap)
            out_frame = Visualizer(alpha=1, cmap=cmap).visualize(arr, results)
            out.write(out_frame)

        # Release resources
        cap.release()
        out.release()

        # Fix with ffmpeg
        fix_video(output_video_path, True)

        # Return processed video
        return output_video_path


@PluginBase.router.post("/vid2densepose", tags=["Video Generation"])
async def vid2densepose(req: Vid2DensePoseRequest):
    plugin: Vid2DensePosePlugin = None
    try:
        plugin = await use_plugin(Vid2DensePosePlugin)
        file_path = plugin.generate(req.video)
        return FileResponse(
            file_path, media_type="video/mp4", filename=os.path.basename(file_path)
        )
    except Exception as e:
        logging.error(f"Error in vid2densepose: {str(e)}")
        return {"error": str(e)}
    finally:
        if plugin is not None:
            release_plugin(plugin)


@PluginBase.router.get("/vid2densepose", tags=["Video Generation"])
async def vid2densepose_from_url(req: Vid2DensePoseRequest = Depends()):
    return await vid2densepose(req)