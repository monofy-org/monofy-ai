import os
from huggingface_hub import snapshot_download
from transformers import WhisperForConditionalGeneration


MODEL_NAME = "openai/whisper-large"

friendly_name = "whisper"
model = None
model_name: str = None


def load_model(model_name=MODEL_NAME):
    path = "models/tts/models--" + MODEL_NAME.replace("/", "--")
    if os.path.isdir(path):
        model_path = os.path.abspath(path)
    else:
        model_path = snapshot_download(
            repo_id=MODEL_NAME, cache_dir="models/whisper", local_dir=path
        )
    if model is None:
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        model.config.forced_decoder_ids = None
        model = model
        model_name = model_name


def offload(for_task):
    model.reset_memory_hooks_state()


def generate():
    # TODO
    pass
