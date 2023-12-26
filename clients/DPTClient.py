import logging
import os
from huggingface_hub import snapshot_download


logging.basicConfig(level=logging.INFO)

MODEL_NAME = "Intel/dpt-large"

friendly_name = "dpt"
model = None
model_name: str = None


def load_model(model_name=MODEL_NAME):
    path = "models/tts/models--" + MODEL_NAME.replace("/", "--")
    if os.path.isdir(path):
        model_path = os.path.abspath(path)
    else:
        model_path = snapshot_download(
            repo_id=MODEL_NAME, cache_dir="models/dpt", local_dir=path
        )
    if model is None:
        model = None  # TODO

        model = model
        model_name = model_name


def offload(for_task):
    # TODO
    pass


def generate():
    # TODO
    pass
