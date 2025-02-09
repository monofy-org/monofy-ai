import os
from colorama import init
from fastapi.responses import FileResponse
import huggingface_hub
import omegaconf
from pydantic import BaseModel
from modules.plugins import PluginBase, use_plugin
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS
from submodules.MagicAnimate.demo.animate_dist import MagicAnimate
from utils.file_utils import cached_snapshot, random_filename
from PIL import Image
from omegaconf import OmegaConf

from utils.stable_diffusion_utils import get_model

class MagicAnimateConfig(BaseModel):
    config: str = "submodules/MagicAnimate/configs/prompts/animation.yaml"    
    rank: int = 0
    world_size: int = 1

class Vid2VidMagicAnimatePlugin(PluginBase):
    name = "MagicAnimate"
    description = "Motion transfer with magic animate"
    instance = None

    def __init__(self):
        super().__init__()

        config = MagicAnimateConfig()
        print(config)

        # model_path = os.path.expanduser("~/.cache/huggingface/hub/models--emilianJR--epiCRealism")
        # if not os.path.exists(model_path):
        #     huggingface_hub.snapshot_download("emilianJR/epiCRealism", local_dir_use_symlinks=False, allow_patterns=["*.safetensors", "*.json"])
        # model_path = os.path.join(model_path, "snapshots")
        # # get first subfolder
        # model_path = os.path.join(model_path, os.listdir(model_path)[0])

        controlnet_model_path = cached_snapshot("zcxu-eric/MagicAnimate")

        conf = OmegaConf.load(config.config)
        conf.inference_config = "submodules/MagicAnimate/configs/inference/inference.yaml"
        conf.pretrained_model_path = cached_snapshot("emilianJR/epiCRealism", allow_patterns=["*.bin", "*.json", "*.txt"])
        conf.pretrained_vae_path = cached_snapshot("stabilityai/sd-vae-ft-mse", allow_patterns=["*.bin", "*.json", "*.txt"])
        conf.pretrained_appearance_encoder_path = os.path.join(            
            controlnet_model_path, "appearance_encoder"
        )
        conf.pretrained_controlnet_path = os.path.join(
            controlnet_model_path, "densepose_controlnet"
        )
        OmegaConf.save(conf, config.config)

        pipeline: MagicAnimate = MagicAnimate(config)

        self.resources["pipeline"] = pipeline

    def animate(
        self,
        image: Image,
        motion_sequence,
        random_seed,
        step,
        guidance_scale,        
    ):
        pipeline: MagicAnimate = self.resources["pipeline"]
        save_path=random_filename("mp4")
        pipeline.predict(image, motion_sequence, random_seed, step, guidance_scale, save_path)

        return FileResponse(save_path)
