import logging
import os
from clients.Singleton import Singleton
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration


logging.basicConfig(level=logging.INFO)

MODEL_NAME = "openai/whisper-large"


class WhisperClient(Singleton):
    def __init__(self):
        super().__init__()
        self.friendly_name = "whisper"
        self.model = None
        self.model_name: str = None

    def load_model(self, model_name=MODEL_NAME):
        path = "models/tts/models--" + MODEL_NAME.replace("/", "--")
        if os.path.isdir(path):
            self.model_path = os.path.abspath(path)
        else:
            self.model_path = snapshot_download(
                repo_id=MODEL_NAME, cache_dir="models/whisper", local_dir=path
            )
        if self.model is None:
            model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
            model.config.forced_decoder_ids = None
            self.model = model
            self.model_name = model_name

    def offload(self, for_task):
        self.model.reset_memory_hooks_state()

    def generate():
        # TODO
        pass
