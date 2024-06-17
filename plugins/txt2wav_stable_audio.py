import logging
import os
from typing import Optional
from fastapi import BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from torch import Tensor
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.file_utils import delete_file, random_filename
from utils.gpu_utils import random_seed_number


class Txt2WavRequest(BaseModel):
    prompt: str
    seconds_start: Optional[int] = 0
    seconds_total: Optional[int] = 30
    seed: Optional[int] = -1


class Txt2WavStableAudioPlugin(PluginBase):

    name = "txt2wav_stable_audio"
    description = "Text-to-wav using Stable Audio"
    instance = None

    def __init__(self):

        super().__init__()

        from stable_audio_tools import get_pretrained_model

        # Download model
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.sample_rate = model_config["sample_rate"]
        self.sample_size = model_config["sample_size"]

        model = model.to(self.device)

        self.resources["model"] = model

    def generate(self, req: Txt2WavRequest):

        import torch
        import torchaudio
        from einops import rearrange
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        conditioning = [
            {
                "prompt": req.prompt,
                "seconds_start": req.seconds_start,
                "seconds_total": req.seconds_total,
            }
        ]

        if req.seed == -1:
            req.seed = random_seed_number()

        model = self.resources["model"]

        # Generate stereo audio
        output = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=self.sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=self.device,
            seed=req.seed,
        )

        # Rearrange audio batch to a single sequence
        output: Tensor = rearrange(output, "b d n -> d (b n)")

        # Peak normalize, clip, convert to int16, and save to file
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )
        
        print(output.shape)

        path = random_filename("wav")

        torchaudio.save(path, output, self.sample_rate)

        return path


@PluginBase.router.post("/txt2wav/stable-audio")
async def txt2wav_stable_audio(background_tasks: BackgroundTasks, req: Txt2WavRequest):
    plugin: Txt2WavStableAudioPlugin = None
    try:
        plugin = await use_plugin(Txt2WavStableAudioPlugin)
        path = plugin.generate(req)
        return FileResponse(path, media_type="audio/wav")
    except Exception as e:
        logging.exception(e)
        raise e
    finally:
        if plugin is not None:
            release_plugin(plugin)
        if os.path.exists(path):
            pass
            #background_tasks.add_task(delete_file(path))


@PluginBase.router.get("/txt2wav/stable-audio")
async def txt2wav_stable_audio_get(req: Txt2WavRequest = Depends()):
    return await txt2wav_stable_audio(req)
