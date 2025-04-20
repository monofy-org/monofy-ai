import logging
import os
from typing import Optional

import torch
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from omegaconf import OmegaConf
from pydantic import BaseModel
import tqdm
import tqdm.rich

from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.video_plugin import VideoPlugin
from settings import USE_XFORMERS
from utils.audio_utils import get_audio_from_request
from utils.console_logging import log_recycle
from utils.file_utils import (
    cached_snapshot,
    random_filename,
)
from utils.video_utils import get_video_from_request


class Vid2VidLatentSyncRequest(BaseModel):
    video: str
    audio: str
    guidance_scale: Optional[float] = 1.0
    num_inference_steps: Optional[int] = 20
    num_frames: Optional[int] = 16
    size: Optional[int] = 256


class Vid2VidLatentSyncPlugin(VideoPlugin):
    name = "LatentSync"
    description = "Image to video generation using LatentSync"
    instance = None

    def __init__(self):
        # self.unet_config_path = None
        # self.inference_ckpt_path = None
        super().__init__()

    def load_model(self):
        from diffusers import AutoencoderKL, DDIMScheduler
        from submodules.LatentSync.latentsync.models.unet import UNet3DConditionModel
        from submodules.LatentSync.latentsync.pipelines.lipsync_pipeline import (
            LipsyncPipeline,
        )

        config = OmegaConf.load("./submodules/LatentSync/configs/unet/first_stage.yaml")

        if config.model.cross_attention_dim == 768:
            self.whisper_model_name = "small.en"
        elif config.model.cross_attention_dim == 384:
            self.whisper_model_name = "tiny.en"
        else:
            config.model.cross_attention_dim == 384
            logging.error("cross_attention_dim must be 768 or 384")

        from submodules.LatentSync.latentsync.whisper.audio2feature import Audio2Feature

        audio_encoder = self.resources.get("audio_encoder")
        if not audio_encoder:
            audio_encoder = Audio2Feature(
                model_path=self.whisper_model_name,
                device="cuda",
                num_frames=config.data.num_frames,
            )
            self.resources["audio_encoder"] = audio_encoder

        vae = self.resources.get("vae")
        if not vae:
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
            )
            vae.config.scaling_factor = 0.18215
            vae.config.shift_factor = 0
            self.resources["vae"] = vae

        unet: UNet3DConditionModel = self.resources.get("unet")
        if not unet:
            # self.unet_config_path = "./submodules/LatentSync/configs/unet/second_stage.yaml"
            # self.inference_ckpt_path = (
            #     "./submodules/LatentSync/checkpoints/latentsync_unet.pt"
            # )

            model_path = cached_snapshot("chunyu-li/LatentSync")

            unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(config.model),
                os.path.join(model_path, "latentsync_unet.pt"),
                device=self.device,
            )
            unet = unet.to(dtype=torch.float16)
            self.resources["unet"] = unet
            if USE_XFORMERS:
                unet.enable_xformers_memory_efficient_attention()

        scheduler: DDIMScheduler = self.resources.get("scheduler")
        if not scheduler:
            scheduler = DDIMScheduler.from_pretrained("./submodules/LatentSync/configs")
            self.resources["scheduler"] = scheduler

        pipeline: LipsyncPipeline = self.resources.get("pipeline")
        if not pipeline:
            pipeline = LipsyncPipeline(
                vae=vae,
                audio_encoder=audio_encoder,
                unet=unet,
                scheduler=scheduler,
            ).to("cuda")
            
            pipeline.progress_bar = tqdm.rich.tqdm

        self.resources["pipeline"] = pipeline
        self.resources["config"] = config
        return pipeline

    async def generate(
        self,
        req: Vid2VidLatentSyncRequest,
    ):        
        video_path = get_video_from_request(req.video)
        audio_path = get_audio_from_request(req.audio)                

        pipeline = self.load_model()
        config = self.resources["config"]

        video_out_path = random_filename("mp4")

        print("DEBUG: default settings (not used)",
            dict(
                num_inference_steps=req.num_inference_steps,
                num_frames=req.num_frames,
                width=req.size,
                height=req.size,
            )
        )

        pipeline(
            video_path=video_path,
            audio_path=audio_path,            
            video_out_path=video_out_path,
            video_mask_path=video_out_path.replace(".mp4", "_mask.mp4"),
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            weight_dtype=torch.float16,
            width=req.size,
            height=req.size,
        )

        if os.path.exists(video_out_path):
            return video_out_path
        else:
            raise Exception("Failed to generate video")


@PluginBase.router.post(
    "/vid2vid/latentsync", tags=["Video Generation (image-to-video)"]
)
async def img2vid_latentsync(req: Vid2VidLatentSyncRequest):
    plugin: Vid2VidLatentSyncPlugin = None
    filename = None
    try:
        plugin: Vid2VidLatentSyncPlugin = await use_plugin(Vid2VidLatentSyncPlugin)
        filename = await plugin.generate(req)
        if not filename:
            raise Exception("Failed to generate video")

        return FileResponse(filename, filename=os.path.basename(filename))
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            release_plugin(Vid2VidLatentSyncPlugin)
        # if filename:
        #     os.remove(full_path)


@PluginBase.router.get(
    "/vid2vid/latentsync", tags=["Video Generation (image-to-video)"]
)
async def img2vid_latentsync_get(req: Vid2VidLatentSyncRequest = Depends()):
    return await img2vid_latentsync(req)
