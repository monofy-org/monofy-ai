import logging
import os
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
from clients.Singleton import Singleton
from utils.gpu_utils import free_vram


class AudioGenClient(Singleton):
    def __init__(self):
        super().__init__()
        self.friendly_name = "audiogen"
        self.model = None

    def generate(
        self,
        prompt: str,
        file_path: str,
        duration: int = 3,
        temperature: float = 1.0,
        cfg_coef=3,
    ):
        free_vram("audiogen", AudioGenClient())

        if self.model is None:
            self.model = AudioGen.get_pretrained("facebook/audiogen-medium")

        self.model.set_generation_params(
            duration=duration, temperature=temperature, cfg_coef=cfg_coef
        )
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

        return os.path.abspath(f"{file_path}.wav")

    def unload(self):
        logging.info(f"Unloading {self.friendly_name}...")
        del self.model

    def offload(self, for_task):
        logging.warn(f"No offload available for {self.friendly_name}.")
        self.unload()
