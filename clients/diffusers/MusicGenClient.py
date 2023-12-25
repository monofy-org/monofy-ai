import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from utils.gpu_utils import free_vram


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

    def generate(
        self,
        prompt: str,
        file_path: str,
        duration: int = 8,
        temperature: float = 1.0,
        cfg_coef=3,
    ):
        free_vram("musicgen", MusicGenClient.instance)

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

    def offload():
        pass