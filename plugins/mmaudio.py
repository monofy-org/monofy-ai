import logging
import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
import torch
import torchaudio
from pydantic import BaseModel

from modules.plugins import PluginBase, release_plugin, router, use_plugin
from utils.console_logging import log_generate, log_loading
from utils.file_utils import random_filename
from utils.gpu_utils import autodetect_dtype, set_seed
from utils.video_utils import get_video_from_request


class MMAudioRequest(BaseModel):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    video: Optional[str] = None
    seed: Optional[int] = -1
    guidance_scale: Optional[float] = 4.5
    num_inference_steps: Optional[int] = 25
    length: Optional[float] = 8.0
    audio_only: Optional[bool] = False


class MMAudioPlugin(PluginBase):
    name = "MMAudio"
    description = "Generate synchronized audio from text or video"
    instance = None

    def __init__(self):
        super().__init__()
        self.resources["model"] = None
        self.dtype = autodetect_dtype(True)

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            logging.warning("CUDA/MPS are not available, running on CPU")

    def offload(self):
        # Offloads already, this is just so it doesn't get unloaded completely by default
        pass

    def load_model(self):
        if self.resources.get("model"):
            return self.resources["model"]

        from submodules.MMAudio.mmaudio.eval_utils import ModelConfig
        from submodules.MMAudio.mmaudio.model.networks import MMAudio, get_my_mmaudio
        from submodules.MMAudio.mmaudio.model.utils.features_utils import FeaturesUtils

        model_name = "small_16k"

        match model_name:
            case "small_16k":
                model = ModelConfig(
                    model_name="small_16k",
                    model_path=Path("./models/mmaudio/weights/mmaudio_small_16k.pth"),
                    vae_path=Path("./models/mmaudio/ext_weights/v1-16.pth"),
                    bigvgan_16k_path=Path("./models/mmaudio/ext_weights/best_netG.pt"),
                    mode="16k",
                    synchformer_ckpt="./models/mmaudio/ext_weights/synchformer_state_dict.pth",
                )
            case "small_44k":
                model = ModelConfig(
                    model_name="small_44k",
                    model_path=Path("./models/mmaudio/weights/mmaudio_small_44k.pth"),
                    vae_path=Path("./models/mmaudio/ext_weights/v1-44.pth"),
                    bigvgan_16k_path=None,
                    mode="44k",
                    synchformer_ckpt="./models/mmaudio/ext_weights/synchformer_state_dict.pth",
                )
            case "medium_44k":
                model = ModelConfig(
                    model_name="medium_44k",
                    model_path=Path("./models/mmaudio/weights/mmaudio_medium_44k.pth"),
                    vae_path=Path("./models/mmaudio/ext_weights/v1-44.pth"),
                    bigvgan_16k_path=None,
                    mode="44k",
                    synchformer_ckpt="./models/mmaudio/ext_weights/synchformer_state_dict.pth",
                )
            case "large_44k":
                model = ModelConfig(
                    model_name="large_44k",
                    model_path=Path("./models/mmaudio/weights/mmaudio_large_44k.pth"),
                    vae_path=Path("./models/mmaudio/ext_weights/v1-44.pth"),
                    bigvgan_16k_path=None,
                    mode="44k",
                )
            case "large_44k_v2":
                model = ModelConfig(
                    model_name="large_44k_v2",
                    model_path=Path(
                        "./models/mmaudio/weights/mmaudio_large_44k_v2.pth"
                    ),
                    vae_path=Path("./models/mmaudio/ext_weights/v1-44.pth"),
                    bigvgan_16k_path=None,
                    mode="44k",
                    synchformer_ckpt="./models/mmaudio/ext_weights/synchformer_state_dict.pth",
                )
            case _:
                raise ValueError(f"Unknown model name: {model_name}")

        log_loading("MMAudio model", model.model_name)

        if (
            not os.path.exists(model.model_path)
            or not os.path.exists(model.vae_path)
            or not os.path.exists(model.synchformer_ckpt)
        ):
            model.download_if_needed()

        # load a pretrained model
        net: MMAudio = (
            get_my_mmaudio(model.model_name).to(self.device, self.dtype).eval()
        )
        net.load_weights(
            torch.load(model.model_path, map_location=self.device, weights_only=True)
        )
        logging.info(f"Loaded weights from {model.model_path}")

        feature_utils = FeaturesUtils(
            tod_vae_ckpt=model.vae_path,
            synchformer_ckpt=model.synchformer_ckpt,
            enable_conditions=True,
            mode=model.mode,
            bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
            need_vae_encoder=False,
        )
        feature_utils = feature_utils.to(self.device, self.dtype).eval()

        self.resources["model"] = model
        self.resources["net"] = net
        self.resources["feature_utils"] = feature_utils
        return model

    def generate(self, req: MMAudioRequest):
        from submodules.MMAudio.mmaudio.eval_utils import (
            ModelConfig,
            generate,
            load_video,
            make_video,
        )
        from submodules.MMAudio.mmaudio.model.utils.features_utils import FeaturesUtils
        from submodules.MMAudio.mmaudio.model.flow_matching import FlowMatching
        from submodules.MMAudio.mmaudio.model.networks import MMAudio

        model: ModelConfig = self.load_model()
        net: MMAudio = self.resources["net"]
        feature_utils: FeaturesUtils = self.resources["feature_utils"]

        seq_cfg = model.seq_cfg

        prompt: str = req.prompt
        negative_prompt: str = req.negative_prompt
        num_steps: int = req.num_inference_steps
        duration: float = req.length
        cfg_strength: float = req.guidance_scale
        skip_video_composite: bool = req.audio_only
        mask_away_clip: bool = False

        _, generator = set_seed(req.seed, True)

        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        if req.video:
            video_path: Path = get_video_from_request(req.video)
        else:
            video_path = None

        if video_path is not None:
            video_info = load_video(video_path, duration)
            clip_frames = video_info.clip_frames
            sync_frames = video_info.sync_frames
            duration = video_info.duration_sec
            if mask_away_clip:
                clip_frames = None
            else:
                clip_frames = clip_frames.unsqueeze(0)
            sync_frames = sync_frames.unsqueeze(0)
        else:
            logging.info("No video provided -- text-to-audio mode")
            clip_frames = sync_frames = None

        seq_cfg.duration = duration

        net.update_seq_lengths(
            seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
        )

        log_generate(f"Generating {duration} seconds of audio...")

        audios = generate(
            clip_frames,
            sync_frames,
            [prompt],
            negative_text=[negative_prompt],
            feature_utils=feature_utils,
            net=net,
            fm=fm,
            rng=generator,
            cfg_strength=cfg_strength,
            clip_batch_size_multiplier=20,
            sync_batch_size_multiplier=20,
        )
        audio = audios.float().cpu()[0]
        audio_filename = random_filename()

        audio_path = f"{audio_filename}.flac"
        torchaudio.save(audio_path, audio, seq_cfg.sampling_rate)

        logging.info(f"Audio saved to {audio_path}")
        output_video: str = None
        if video_path is not None and not skip_video_composite:
            output_video = random_filename("mp4")
            make_video(
                video_info, output_video, audio, sampling_rate=seq_cfg.sampling_rate
            )
            logging.info(f"Video saved to {output_video}")

        return audio_path, output_video


@router.post("/mmaudio")
async def generate_audio(req: MMAudioRequest):
    plugin: MMAudioPlugin = None
    try:
        plugin = await use_plugin(MMAudioPlugin)
        audio, video = plugin.generate(req)

        if req.video is not None and not req.audio_only:
            return FileResponse(
                video,
                media_type="video/mp4",
            )
        else:
            return FileResponse(
                audio,
                media_type="audio/flac",
            )
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            release_plugin(plugin)


@router.get("/mmaudio")
async def get_mmaudio(req: MMAudioRequest = Depends()):
    return await generate_audio(req)
