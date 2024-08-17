from copy import deepcopy
import os
import random
import sys
from typing import Literal

from fastapi import Depends
from fastapi.responses import FileResponse
import numpy as np
from omegaconf import OmegaConf
import torch
from classes.requests import Txt2VidRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from funcs import load_model_checkpoint
from train_t2v_lora import (
    get_parser,
    run_training,
)
from vader_utils import instantiate_from_config
from utils.file_utils import cached_snapshot
from utils.gpu_utils import set_seed
from huggingface_hub import snapshot_download
from pytorch_lightning import seed_everything


def setup_model():
    try:
        parser = get_parser()
        args = parser.parse_args()
    except Exception:
        args = {}

    ## ------------------------step 2: model config-----------------------------
    # download the checkpoint for VideoCrafter2 model
    ckpt_dir = cached_snapshot("VideoCrafter/VideoCrafter2")
    args.ckpt_path = os.path.join(ckpt_dir, "model.ckpt")
    args.config = (
        "submodules/VADER/VADER-VideoCrafter/configs/inference_t2v_512_v2.0.yaml"
    )

    # load the model
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)

    assert os.path.exists(
        args.ckpt_path
    ), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)

    # convert first_stage_model and cond_stage_model to torch.float16 if mixed_precision is True
    if args.mixed_precision != "no":
        model.first_stage_model = model.first_stage_model.half()
        model.cond_stage_model = model.cond_stage_model.half()

    print("Model setup complete!")
    print("model dtype: ", model.dtype)
    return model


def seed_everything_self(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
    
def main_fn(
    prompt,
    lora_model,
    lora_rank,
    seed=200,
    height=320,
    width=512,
    unconditional_guidance_scale=12,
    ddim_steps=25,
    ddim_eta=1.0,
    frames=24,
    savefps=10,
    model=None,
):

    parser = get_parser()
    args = parser.parse_args()

    # overwrite the default arguments
    args.prompt_str = prompt
    args.lora_ckpt_path = lora_model
    args.lora_rank = lora_rank
    args.seed = seed
    args.height = height
    args.width = width
    args.unconditional_guidance_scale = unconditional_guidance_scale
    args.ddim_steps = ddim_steps
    args.ddim_eta = ddim_eta
    args.frames = frames
    args.savefps = savefps

    seed_everything(args.seed)
    seed_everything_self(args.seed)

    print ("Running with args: ", args)

    video_path = run_training(model, args)

    return video_path


class Txt2VidVADERPlugin(PluginBase):
    name = "Txt2Vid (VADER)"
    description = "Text-to-video generation with VADER"
    instance = None

    def __init__(self):
        super().__init__()

        self.resources["model"] = setup_model()

    def generate(self, req: Txt2VidRequest):
        model = self.resources["model"]

        lora_rank: Literal[8, 16] = 16
        lora_model: Literal["huggingface-pickscore", "huggingface-hps-aesthetic"] = (
            "huggingface-pickscore"
        )

        video_path = main_fn(
            prompt=req.prompt,
            lora_model=lora_model,
            lora_rank=lora_rank,
            seed=set_seed(req.seed),
            height=req.height,
            width=req.width,
            unconditional_guidance_scale=req.guidance_scale,
            ddim_steps=req.num_inference_steps,
            ddim_eta=1,
            frames=req.num_frames,
            savefps=req.fps,
            model=deepcopy(model),
        )

        return video_path


@PluginBase.router.post("/txt2vid/vader", tags=["Text-to-Video"])
async def txt2vid_vader(req: Txt2VidRequest):
    plugin: Txt2VidVADERPlugin = None
    try:
        plugin = await use_plugin(Txt2VidVADERPlugin)
        path = plugin.generate(req)
        return FileResponse(path, media_type="video/mp4")

    finally:
        if plugin is not None:
            release_plugin(Txt2VidVADERPlugin)


@PluginBase.router.get("/txt2vid/vader", tags=["Text-to-Video"])
async def txt2vid_vader_from_url(req: Txt2VidRequest = Depends()):
    return await txt2vid_vader(req)
