import logging
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from clients.Singleton import Singleton
from utils.gpu_utils import free_vram


class MusicGenClient(Singleton):
    def __init__(self):
        super().__init__()
        self.friendly_name = "musicgen"
        self.model = None

    def generate(
        self,
        prompt: str,
        file_path: str,
        duration: int = 8,
        temperature: float = 1.0,
        cfg_coef=3,
    ):
        free_vram(self.friendly_name, MusicGenClient())

        if self.model is None:
            self.model = MusicGen.get_pretrained("facebook/musicgen-small")

        self.model.set_generation_params(
            duration=duration, temperature=temperature, cfg_coef=cfg_coef
        )
        wav = self.model.generate([prompt], progress=True)

        for _, one_wav in enumerate(wav):
            audio_write(
                file_path, one_wav.cpu(), self.model.sample_rate, strategy="peak"
            )

        del self.model
        self.model = None

        return os.path.abspath(f"{file_path}.wav")

    def unload(self):
        logging.info(f"Unloading {self.friendly_name}...")
        del self.model

    def offload(self, for_task):
        logging.warn(f"No offload available for {self.friendly_name}.")
        self.unload()
