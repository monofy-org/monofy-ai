import gc
import io
import logging
import os
from threading import Thread
import time
import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from classes.musicgen_streamer import MusicgenStreamer
from clients.ClientBase import ClientBase
from utils.gpu_utils import autodetect_device, load_gpu_task, set_seed
from utils.misc_utils import print_completion_time
from settings import MUSICGEN_MODEL

MUSICGEN_USE_FP16 = False


class MusicGenClient(ClientBase):

    def __init__(self):
        super().__init__("musicgen")

    def load_models(self, model_name=MUSICGEN_MODEL):
        if len(self.models) == 0:
            device = autodetect_device()

            if MUSICGEN_USE_FP16:
                fp16_path = os.path.join("models", model_name + "-fp16")
                converted_exists = os.path.exists(fp16_path)
                if not converted_exists:
                    logging.info(
                        f"Converting {model_name} to FP16... This is a one-time operation."
                    )

            ClientBase.load_model(
                self,
                AutoProcessor,
                model_name,
                allow_fp16=MUSICGEN_USE_FP16,
                allow_bf16=False,
                device=device,
                set_variant_fp16=False,
            )

            ClientBase.load_model(
                self,
                MusicgenForConditionalGeneration,
                fp16_path if (MUSICGEN_USE_FP16 and converted_exists) else model_name,
                unload_previous_model=False,
                allow_fp16=MUSICGEN_USE_FP16,
                allow_bf16=False,
                set_variant_fp16=False,
            )
            self.models[1].to(device)

            if MUSICGEN_USE_FP16 and not converted_exists:
                # self.models[1] = self.models[1].half()
                self.models[1].save_pretrained(fp16_path)
                logging.info(f"Model saved to {fp16_path}")

            self.models.append(
                MusicgenStreamer(self.models[1], device=device, play_steps=100)
            )

    async def generate(
        self,
        prompt: str,
        duration: int = 8,
        temperature: float = 1.05,
        guidance_scale: float = 3.0,
        top_k=250,
        top_p: float = 0.97,
        format: str = "wav",
        wav_bytes: bytes = None,
        seed: int = -1,
        streaming: bool = False,
    ):
        async with load_gpu_task(self.friendly_name, self):

            self.load_models()

            start_time = time.time()

            processor: AutoProcessor = self.models[0]
            model: MusicgenForConditionalGeneration = self.models[1]

            if streaming:
                streamer: MusicgenStreamer = self.models[2] if streaming else None
                streamer.token_cache = None

            sampling_rate = model.config.audio_encoder.sampling_rate

            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
                sampling_rate=sampling_rate,
            ).to(model.device)

            set_seed(seed)

            if wav_bytes is None:

                logging.info(f"Generating {duration}s of music...")

                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=int(duration * 50),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    guidance_scale=guidance_scale,
                )

                max_range = np.iinfo(np.int16).max

                if not streaming:
                    new_audio = (
                        model.generate(**generation_kwargs).unsqueeze(0).cpu().numpy()
                    )
                    print_completion_time(start_time, "musicgen")
                    new_audio = np.clip(
                        new_audio, -1, 1
                    )  # ensure data is within range [-1, 1]
                    new_audio = (new_audio * max_range).astype(np.int16)
                    yield sampling_rate, new_audio
                else:
                    generation_kwargs["streamer"] = streamer
                    thread = Thread(target=model.generate, kwargs=generation_kwargs)
                    thread.start()
                    for new_audio in streamer:
                        print(
                            f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 2)} seconds"
                        )
                        new_audio = np.clip(
                            new_audio, -1, 1
                        )  # ensure data is within range [-1, 1]
                        new_audio = (new_audio * max_range).astype(np.int16)
                        if new_audio.shape[0] > 0:
                            yield sampling_rate, new_audio
            else:

                logging.info("Generating continuation...")

                tensor, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))

                new_audio = model.generate_continuation(
                    tensor,
                    sample_rate,
                    [prompt],
                    max_new_tokens=int(duration * 50),
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    guidance_scale=guidance_scale,
                    streamer=streamer,
                )

            print_completion_time(start_time, "musicgen")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __del__(self):
        self.unload()
