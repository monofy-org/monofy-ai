import io
import logging
from typing import Optional
from fastapi import BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from torch import Tensor
import torch
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.gpu_utils import clear_gpu_cache, random_seed_number


class Txt2WavRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    seconds_start: Optional[int] = 0
    seconds_total: Optional[int] = 30
    seed: Optional[int] = -1
    guidance_scale: Optional[float] = 7.0
    num_inference_steps: Optional[int] = 100


class Txt2WavStableAudioPlugin(PluginBase):

    name = "Stable Audio"
    description = "Text-to-wav using Stable Audio"
    instance = None

    def __init__(self):

        super().__init__()

        from stable_audio_tools import get_pretrained_model

        # Download model
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        self.sample_rate = model_config["sample_rate"]
        self.max_sample_size = model_config["sample_size"]

        model = model.to(self.device)

        self.resources["model"] = model
        self.resources["config"] = model_config

    def generate(self, req: Txt2WavRequest):

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

        kwargs = dict(
            steps=req.num_inference_steps,
            cfg_scale=req.guidance_scale,
            conditioning=conditioning,
            sample_size=min(req.seconds_total * self.sample_rate, self.max_sample_size),
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=self.device,
            seed=req.seed,
        )

        if req.negative_prompt:
            kwargs["negative_conditioning"] = [
                {
                    "prompt": req.negative_prompt,
                    "seconds_start": req.seconds_total,
                    "seconds_total": 0,
                }
            ]

        # Generate stereo audio
        output = generate_diffusion_cond(model, **kwargs)

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

        wav_io = io.BytesIO()
        torchaudio.save(wav_io, output, self.sample_rate, format="wav")

        return self.sample_rate, wav_io.getvalue()


@PluginBase.router.post("/txt2wav/stable-audio")
async def txt2wav_stable_audio(background_tasks: BackgroundTasks, req: Txt2WavRequest):
    plugin: Txt2WavStableAudioPlugin = None
    try:
        plugin = await use_plugin(Txt2WavStableAudioPlugin)
        sample_rate, data = plugin.generate(req)
        return StreamingResponse(
            io.BytesIO(data),
            media_type="audio/wav",
            headers={"Sample-Rate": str(sample_rate)},
        )
    except Exception as e:
        logging.exception(e)
        raise e
    finally:
        if plugin is not None:
            release_plugin(plugin)
            clear_gpu_cache()


@PluginBase.router.get("/txt2wav/stable-audio")
async def txt2wav_stable_audio_get(req: Txt2WavRequest = Depends()):
    return await txt2wav_stable_audio(req)
