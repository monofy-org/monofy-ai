import io
import logging
import torchaudio
import numpy as np
from threading import Thread
from pydantic import BaseModel
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from classes.musicgen_streamer import MusicgenStreamer
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.audio_utils import get_audio_loop, wav_io
from utils.gpu_utils import autodetect_dtype, set_seed
from settings import MUSICGEN_MODEL


MAX_INT16 = np.iinfo(np.int16).max


class MusicGenRequest(BaseModel):
    prompt: str
    duration: int = 10
    temperature: float = 1.0
    guidance_scale: float = 6.5
    format: str = "wav"
    seed: int = -1
    top_p: float = 0.6
    streaming: bool = False
    wav_bytes: str | None = None
    loop: bool = False


class MusicGenPlugin(PluginBase):

    name = "MusicGen"
    description = "Music generation"
    instance = None

    def __init__(self):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        super().__init__()

        self.dtype = autodetect_dtype(bf16_allowed=False)

        processor: AutoProcessor = AutoProcessor.from_pretrained(
            MUSICGEN_MODEL, dtype=self.dtype
        )

        model: MusicgenForConditionalGeneration = (
            MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL).to(
                self.device, dtype=self.dtype
            )
        ).half()

        streamer = MusicgenStreamer(model, play_steps=50) # 50 = 1 second

        self.resources = {"model": model, "processor": processor, "streamer": streamer}

    def generate(
        self,
        req: MusicGenRequest,
    ):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        model: MusicgenForConditionalGeneration = self.resources["model"]
        processor: AutoProcessor = self.resources["processor"]

        sampling_rate: int = model.config.audio_encoder.sampling_rate

        if req.streaming:
            streamer: MusicgenStreamer = self.resources["streamer"]
            streamer.token_cache = None

        set_seed(req.seed)

        if req.wav_bytes is None:

            inputs = processor(
                text=[req.prompt],
                padding=False,
                return_tensors="pt",
                sampling_rate=sampling_rate,
            ).to(model.device)

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=int(req.duration * 50),
                temperature=req.temperature,
                # top_k=req.top_k,
                top_p=req.top_p,
                guidance_scale=req.guidance_scale,
            )

            max_range = np.iinfo(np.int16).max

            if req.streaming is False:
                logging.info(f"Generating {req.duration}s of music...")

                generated_audio = model.generate(**generation_kwargs)
                new_audio = generated_audio.unsqueeze(0).cpu().numpy()[0][0][0]

                output = np.clip(
                    new_audio, -1, 1
                )  # ensure data is within range [-1, 1]
                output = (output * max_range).astype(np.int16)
                yield sampling_rate, output
            else:
                logging.info(f"Streaming {req.duration}s of music...")
                generation_kwargs["streamer"] = streamer
                thread = Thread(
                    target=model.generate, kwargs=generation_kwargs, daemon=True
                )
                thread.start()
                for new_audio in streamer:
                    print(
                        f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 2)} seconds"
                    )
                    output = np.clip(
                        new_audio, -1, 1
                    )  # ensure data is within range [-1, 1]
                    output = (output * max_range).astype(np.int16)
                    if new_audio.shape[0] > 0:
                        yield sampling_rate, output
        else:

            logging.info("Generating continuation...")

            tensor, sample_rate = torchaudio.load(io.BytesIO(req.wav_bytes))

            new_audio = self.pipeline.generate_continuation(
                tensor,
                sample_rate,
                [req.prompt],
                max_new_tokens=int(req.duration * 50),
                temperature=req.temperature,
                top_p=req.top_p,
                guidance_scale=req.guidance_scale,
                # streamer=streamer,
            )
            new_audio = np.clip(new_audio, -1, 1)  # ensure data is within range [-1, 1]
            new_audio = (new_audio * max_range).astype(np.int16)

            yield sampling_rate, new_audio


@PluginBase.router.post("/musicgen", response_class=StreamingResponse)
async def musicgen(req: MusicGenRequest):
    plugin = None
    try:
        plugin: MusicGenPlugin = await use_plugin(MusicGenPlugin)
        sampling_rate, wav_bytes = next(plugin.generate(req))
        wave_bytes_io = wav_io(wav_bytes, sampling_rate, req.format)
        if req.loop:
            wave_bytes_io = get_audio_loop(wave_bytes_io)
        return StreamingResponse(wave_bytes_io, media_type="audio/" + req.format)

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin is not None:
            release_plugin(MusicGenPlugin)


@PluginBase.router.get("/musicgen", response_class=StreamingResponse)
async def musicgen_get(
    req: MusicGenRequest = Depends(),
):
    return await musicgen(req)
