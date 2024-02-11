import gc
import io
import logging
import time
import numpy as np
import torch
import torchaudio
from clients.ClientBase import ClientBase
from audiocraft.models import MusicGen
from scipy.io.wavfile import write
from utils.gpu_utils import load_gpu_task, set_seed
from utils.misc_utils import print_completion_time
from settings import MUSICGEN_MODEL


class MusicGenClient(ClientBase):    

    def __init__(self):
        super().__init__("musicgen")
        self.models: list[MusicGen] = []

    def load_models(self, model_name=MUSICGEN_MODEL):
        if len(self.models) == 0:
            ClientBase.load_model(
                self, MusicGen, model_name, allow_fp16=False, allow_bf16=False
            )

    def generate(
        self,
        prompt: str,
        duration: int = 8,
        temperature: float = 1.0,
        guidance_scale: float = 3.0,
        top_p: float = 0.9,
        format: str = "wav",
        wav_bytes: bytes = None,
        seed: int = -1,
    ) -> bytes:
        load_gpu_task(self.friendly_name, self)

        if len(self.models) == 0:
            self.load_models()

        start_time = time.time()

        set_seed(seed)

        model = self.models[0]

        model.set_generation_params(
            duration=duration, temperature=temperature, cfg_coef=guidance_scale, top_k=250, top_p=top_p
        )

        if wav_bytes is None:

            logging.info(f"Generating {duration}s of music...")

            wav = model.generate(
                descriptions=[prompt],
            )
        else:

            logging.info("Generating continuation...")

            tensor, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))                        
            
            print(model)

            wav = model.generate_continuation(
                tensor,
                sample_rate,
                [prompt],
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
