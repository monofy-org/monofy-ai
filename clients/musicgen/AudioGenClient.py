import os
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
from utils.gpu_utils import free_vram


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

    def generate(
        self, prompt: str, file_path: str, duration: int = 3, temperature: float = 1.0
    ):
        free_vram("audiogen")

        if self.model is None:
            self.model = AudioGen.get_pretrained("facebook/audiogen-medium")

        self.model.set_generation_params(duration=duration, temperature=temperature)
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
        self.model = None

        return os.path.abspath(file_path)
