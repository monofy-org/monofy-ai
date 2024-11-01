import io
import logging
import torchaudio
import wave
import numpy as np
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from classes.requests import MusicGenRequest
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.audio_utils import get_audio_loop, wav_to_mp3
from utils.console_logging import log_generate, log_loading
from utils.file_utils import random_filename
from utils.gpu_utils import (
    autodetect_device,
    autodetect_dtype,
    set_seed,
)
from settings import MUSICGEN_MODEL


MAX_INT16 = np.iinfo(np.int16).max


class Txt2WavMusicGenPlugin(PluginBase):
    name = "MusicGen"
    description = "Music generation"
    instance = None

    def __init__(self):
        self.use_audiocraft = False

        # from submodules.audiocraft.audiocraft.models import MusicGen
        from transformers import MusicgenForConditionalGeneration

        super().__init__()

        self.device = autodetect_device(allow_accelerate=True)
        self.dtype = autodetect_dtype(allow_bf16=False)

        log_loading("audio model", MUSICGEN_MODEL)

        # musicgen: MusicGen = MusicGen.get_pretrained(MUSICGEN_MODEL)
        self.resources["model"] = MusicgenForConditionalGeneration.from_pretrained(
            MUSICGEN_MODEL,
        ).to(self.device)

        if not self.use_audiocraft:
            from transformers import AutoProcessor

            self.resources["processor"] = AutoProcessor.from_pretrained(
                MUSICGEN_MODEL, device=self.device
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

        # model = MusicgenForConditionalGeneration.from_pretrained(
        #     MUSICGEN_MODEL, torch_dtype=self.dtype
        # ).to(self.device)

        # from classes.musicgen_streamer import MusicgenStreamer

        # streamer = MusicgenStreamer(
        #     model, device=self.device, play_steps=50
        # )  # 50 = 1 second

    def generate(
        self,
        req: MusicGenRequest,
    ):
        # from submodules.audiocraft.audiocraft.models import MusicGen
        # model: MusicGen = self.resources["model"]
        from transformers import PreTrainedModel

        model: PreTrainedModel = self.resources["model"]

        sampling_rate: int = model.config.audio_encoder.sampling_rate

        if req.streaming:
            from classes.musicgen_streamer import MusicgenStreamer

            streamer: MusicgenStreamer = self.resources["streamer"]
            streamer.token_cache = None

        set_seed(req.seed)

        if req.wav_bytes is None:
            if self.use_audiocraft:
                model.set_generation_params(
                    True,
                    0,
                    req.top_p,
                    req.temperature,
                    req.duration,
                    req.guidance_scale,
                )

            from tqdm.rich import tqdm

            estimated_steps = 50 * req.duration
            with tqdm(
                total=estimated_steps,
                unit="%",
                dynamic_ncols=True,
                position=0,
                leave=True,
                delay=1.0,
            ) as pbar:

                def update_progress(frame, total_frames):
                    if frame > 1:
                        progress = round((frame / total_frames) * estimated_steps)
                        pbar.update(progress - pbar.n)

                args = {}

                if self.use_audiocraft:
                    model.set_custom_progress_callback(update_progress)
                    args["progress"] = True
                    new_audio = model.generate([req.prompt], **args)
                else:
                    processor = self.resources["processor"]
                    args["do_sample"] = True
                    args["guidance_scale"] = req.guidance_scale
                    args["max_new_tokens"] = 50 * (req.duration or 10)
                    inputs = processor(
                        text=[req.prompt],
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                    new_audio = model.generate(**inputs, **args)

                yield sampling_rate, new_audio

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
            new_audio = (new_audio * MAX_INT16).astype(np.int16)

            yield sampling_rate, new_audio


@PluginBase.router.post(
    "/txt2wav/musicgen", response_class=StreamingResponse, tags=["Audio and Music"]
)
async def musicgen(req: MusicGenRequest):
    plugin = None
    try:
        plugin: Txt2WavMusicGenPlugin = await use_plugin(Txt2WavMusicGenPlugin)

        filename = random_filename(req.format)
        wave_io = io.BytesIO()

        sr = None

        buffers = []

        log_generate(f"Generating {req.duration} seconds of audio...")

        for sampling_rate, data in plugin.generate(req):
            sr = sampling_rate
            buffers.append(data.cpu())

        if sr is None:
            raise Exception("No audio generated")

        sfdata = np.concatenate(buffers, axis=1).squeeze().clip(-1, 1)

        if req.loop:
            try:
                wave_io = get_audio_loop(sfdata, sampling_rate)
            except Exception as e:
                logging.warning(e, exc_info=True)
                logging.warning("Could not loop", exc_info=True)
        else:
            channels = 2 if "stereo" in MUSICGEN_MODEL else 1
            with wave.open(wave_io, "wb") as wav_file:
                wav_file.setnchannels(channels)  # Mono audio
                wav_file.setsampwidth(2)  # 16-bit audio
                wav_file.setframerate(sr)
                wav_file.writeframes((sfdata * 32767).astype(np.int16).tobytes())
            wave_io.seek(0)

        if req.format == "mp3":
            assert sr is not None

            wave_io = wav_to_mp3(wave_io, sr)

            return StreamingResponse(
                wave_io,
                media_type=f"audio/{req.format}",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        else:
            return StreamingResponse(
                wave_io,
                media_type=f"audio/{req.format}",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin is not None:
            release_plugin(Txt2WavMusicGenPlugin)


@PluginBase.router.get(
    "/txt2wav/musicgen", response_class=StreamingResponse, tags=["Audio and Music"]
)
async def musicgen_get(
    req: MusicGenRequest = Depends(),
):
    return await musicgen(req)
