import io
import logging
from typing import Optional
from fastapi import BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import numpy as np
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.gpu_utils import clear_gpu_cache, random_seed_number
from diffusers import StableAudioPipeline


class Txt2WavRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    seconds_start: Optional[int] = 0
    seconds_total: Optional[int] = 47.55446712018141  # Max for Stable Audio is 47.55446712018141 seconds
    seed: Optional[int] = -1
    guidance_scale: Optional[float] = 7.0
    num_inference_steps: Optional[int] = 100


class Txt2WavStableAudioPlugin(PluginBase):

    name = "Stable Audio"
    description = "Text-to-wav using Stable Audio"
    instance = None

    def __init__(self):
        super().__init__()
        
        pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            torch_dtype=torch.float16
        )
        pipe = pipe.to(self.device)
        
        self.resources["pipe"] = pipe
        self.sample_rate = pipe.vae.sampling_rate

    async def generate(self, req: Txt2WavRequest):
        pipe = self.resources["pipe"]

        if req.seed == -1:
            req.seed = random_seed_number()

        generator = torch.Generator(self.device).manual_seed(req.seed)

        audio = pipe(
            req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            audio_end_in_s=float(req.seconds_total),
            guidance_scale=req.guidance_scale,
            generator=generator,
        ).audios

        output = audio[0].T.float().cpu().numpy()
        
        wav_io = io.BytesIO()
        import soundfile as sf
        sf.write(wav_io, output, self.sample_rate, format='WAV')
        
        return self.sample_rate, wav_io.getvalue()


@PluginBase.router.post("/txt2wav/stable-audio", tags=["Audio and Music"])
async def txt2wav_stable_audio(background_tasks: BackgroundTasks, req: Txt2WavRequest):
    plugin: Txt2WavStableAudioPlugin = None
    try:
        plugin = await use_plugin(Txt2WavStableAudioPlugin)
        sample_rate, data = await plugin.generate(req)
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


@PluginBase.router.get("/txt2wav/stable-audio", tags=["Audio and Music"])
async def txt2wav_stable_audio_get(req: Txt2WavRequest = Depends()):
    return await txt2wav_stable_audio(req)