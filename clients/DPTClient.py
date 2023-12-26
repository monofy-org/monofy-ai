import logging
import os
from clients.Singleton import Singleton
from huggingface_hub import snapshot_download


logging.basicConfig(level=logging.INFO)

MODEL_NAME = "Intel/dpt-large"


class DPTClient(Singleton):
    def __init__(self):
        super().__init__()
        self.friendly_name = "dpt"
        self.model = None
        self.model_name: str = None

    def load_model(self, model_name=MODEL_NAME):
        path = "models/tts/models--" + MODEL_NAME.replace("/", "--")
        if os.path.isdir(path):
            self.model_path = os.path.abspath(path)
        else:
            self.model_path = snapshot_download(
                repo_id=MODEL_NAME, cache_dir="models/dpt", local_dir=path
            )
        if self.model is None:
            model = None  # TODO

            self.model = model
            self.model_name = model_name

    def offload(self, for_task):
        # TODO
        pass

    def generate():
        # TODO
        pass
