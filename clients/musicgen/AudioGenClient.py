import logging
import os
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import torch


class AudioGenClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist
        return cls._instance

    def __init__(self):
        self.model = None
        logging.info("Loading audiogen...")

    def generate(self, prompt: str, file_path: str, duration: int = 3):
        self.model = AudioGen.get_pretrained("facebook/audiogen-medium")
        self.model.set_generation_params(duration=duration)
        wav = self.model.generate([prompt], progress=True)

        for _, one_wav in enumerate(wav):
            audio_write(
                file_path,
                one_wav.cpu(),
                self.model.sample_rate,
                strategy="peak",
                loudness_compressor=True,
            )

        del self.model
        torch.cuda.empty_cache()

        return os.path.abspath(file_path)
