import io
import logging
import os
import torchaudio
import numpy as np
from threading import Thread
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from classes.requests import MusicGenRequest
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.audio_utils import get_audio_loop, wav_io
from utils.console_logging import log_disk, log_loading
from utils.file_utils import random_filename
from utils.gpu_utils import autodetect_dtype, set_seed
from settings import MUSICGEN_MODEL


MAX_INT16 = np.iinfo(np.int16).max


class MusicGenPlugin(PluginBase):
    name = "MusicGen"
    description = "Music generation"
    instance = None

    def __init__(self):
        from transformers import MusicgenProcessor, MusicgenForConditionalGeneration

        super().__init__()

        self.dtype = autodetect_dtype(bf16_allowed=False)

        log_loading("audio model", MUSICGEN_MODEL)

        processor: MusicgenProcessor = MusicgenProcessor.from_pretrained(
            MUSICGEN_MODEL, dtype=self.dtype
        )

        # converted_model_path = os.path.join(
        #     "models", "musicgen", f"{os.path.basename(MUSICGEN_MODEL)}_{self.dtype}"
        # )
        # if not os.path.exists(converted_model_path):
        #     log_disk(f"Converting model to {self.dtype} (one-time operation)...")
        #     model = MusicgenForConditionalGeneration.from_pretrained(
        #         MUSICGEN_MODEL, torch_dtype=self.dtype
        #     ).to(self.device, self.dtype)
        #     model.save_pretrained(converted_model_path)
        # else:
        #     model = MusicgenForConditionalGeneration.from_pretrained(
        #         converted_model_path, torch_dtype=self.dtype
        #     ).to(self.device, self.dtype)

        model = MusicgenForConditionalGeneration.from_pretrained(
            MUSICGEN_MODEL, torch_dtype=self.dtype
        ).to(self.device, self.dtype)

        from classes.musicgen_streamer import MusicgenStreamer

        streamer = MusicgenStreamer(
            model, device=self.device, play_steps=50
        )  # 50 = 1 second

        self.resources = {"model": model, "processor": processor, "streamer": streamer}

    def generate(
        self,
        req: MusicGenRequest,
    ):
        from transformers import MusicgenProcessor, MusicgenForConditionalGeneration

        model: MusicgenForConditionalGeneration = self.resources["model"]
        processor: MusicgenProcessor = self.resources["processor"]

        sampling_rate: int = model.config.audio_encoder.sampling_rate

        if req.streaming:
            from classes.musicgen_streamer import MusicgenStreamer

            streamer: MusicgenStreamer = self.resources["streamer"]
            streamer.token_cache = None

        set_seed(req.seed)

        if req.wav_bytes is None:
            inputs = processor(
                text=[req.prompt],
                padding=True,
                return_tensors="pt",
                sampling_rate=sampling_rate,
            ).to(self.device)

            generation_kwargs = dict(
                **inputs,
                max_new_tokens=int(req.duration * 50),
                temperature=req.temperature,
                # top_k=req.top_k,
                top_p=req.top_p,
                guidance_scale=req.guidance_scale,
            )

            max_range = np.iinfo(np.int16).max

            if req.streaming:
                logging.info(f"Streaming {req.duration}s of music...")
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
                        logging.info("No audio generated, skipping...")
                        yield None, None

            else:
                logging.info(f"Generating {req.duration}s of music...")

                generated_audio = model.generate(**generation_kwargs)
                new_audio = generated_audio.unsqueeze(0).cpu().numpy()[0][0][0]

                output = np.clip(
                    new_audio, -1, 1
                )  # ensure data is within range [-1, 1]
                output = (output * max_range).astype(np.int16)
                yield sampling_rate, output

        else:
            # TODO: this is currently not in use and needs exploration

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


@PluginBase.router.post(
    "/musicgen", response_class=StreamingResponse, tags=["Audio and Music"]
)
async def musicgen(req: MusicGenRequest):
    plugin = None
    try:
        plugin: MusicGenPlugin = await use_plugin(MusicGenPlugin)
        import soundfile as sf

        filename = random_filename(req.format)
        wave_bytes_io = io.BytesIO()

        for sampling_rate, wav_bytes in plugin.generate(req):
            sf.write(
                wave_bytes_io,
                wav_bytes,
                samplerate=sampling_rate,
                format=req.format,
                subtype="PCM_16",
            )
            wave_bytes_io.seek(0)

        if req.loop:
            wave_bytes_io = get_audio_loop(wave_bytes_io)

        return StreamingResponse(
            wave_bytes_io,
            media_type=f"audio/{req.format}",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin is not None:
            release_plugin(MusicGenPlugin)


@PluginBase.router.get(
    "/musicgen", response_class=StreamingResponse, tags=["Audio and Music"]
)
async def musicgen_get(
    req: MusicGenRequest = Depends(),
):
    return await musicgen(req)
