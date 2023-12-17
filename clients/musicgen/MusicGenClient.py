import logging
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch

from utils.torch_utils import autodetect_device


class MusicGenClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist
        return cls._instance

    def __init__(self):                
        self.model = None
        logging.info("Loading musicgen...")                
    def generate(self, prompt: str, name: str, duration: int = 3):
        self.model = MusicGen.get_pretrained("facebook/musicgen-small")
        self.model.set_generation_params(duration=duration)
        wav = self.model.generate([prompt], progress=True)

        for _, one_wav in enumerate(wav):
            audio_write(f"{name}", one_wav.cpu(), self.model.sample_rate, strategy="loudness")        

        del self.model
        torch.cuda.empty_cache()