import gc
import io
import logging
import time
import numpy as np
import torch
import torchaudio
from clients.ClientBase import ClientBase
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io.wavfile import write
from utils.gpu_utils import autodetect_device, load_gpu_task, set_seed
from utils.misc_utils import print_completion_time
from settings import MUSICGEN_MODEL


class MusicGenClient(ClientBase):

    def __init__(self):
        super().__init__("musicgen")

    def load_models(self, model_name=MUSICGEN_MODEL):
        if len(self.models) == 0:
            ClientBase.load_model(
                self, AutoProcessor, model_name, allow_fp16=False, allow_bf16=False
            )
            ClientBase.load_model(
                self,
                MusicgenForConditionalGeneration,
                model_name,
                unload_previous_model=False,
                allow_fp16=False,
                allow_bf16=False,
            )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        duration: int = 8,
        temperature: float = 1.05,
        guidance_scale: float = 3.0,
        top_p: float = 0.95,
        format: str = "wav",
        wav_bytes: bytes = None,
        seed: int = -1,
    ) -> bytes:
        load_gpu_task(self.friendly_name, self)

        if len(self.models) == 0:
            self.load_models()
        
        sampling_rate = self.models[1].config.audio_encoder.sampling_rate        

        start_time = time.time()

        inputs = self.models[0](
            text=[prompt],
            padding=True,
            return_tensors="pt",
            sampling_rate=sampling_rate,
        ).to(autodetect_device())        

        set_seed(seed)

        model: MusicgenForConditionalGeneration = self.models[1]

        if wav_bytes is None:

            logging.info(f"Generating {duration}s of music...")            

            wav = model.generate(
                **inputs,
                max_new_tokens=int(duration * 50),
                temperature=temperature,
                top_p=top_p,
                guidance_scale=guidance_scale,
            )
        else:

            logging.info("Generating continuation...")

            tensor, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))                        

            wav = model.generate_continuation(
                tensor,
                sample_rate,
                [prompt],
                max_new_tokens=int(duration * 50),             
                temperature=temperature,
                top_p=top_p,
                guidance_scale=guidance_scale,
            )

        print_completion_time(start_time, "musicgen")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        wav = wav.cpu().numpy()
        wav = np.clip(wav, -1, 1)  # ensure data is within range [-1, 1]
        wav = (wav * 32767).astype(np.int16)  # scale to int16 range and convert
        wav_bytes = io.BytesIO()
        write(wav_bytes, 32000, wav)

        return wav_bytes.getvalue()

    def __del__(self):
        self.unload()
