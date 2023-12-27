import logging
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from utils.gpu_utils import load_gpu_task
from clients import MusicGenClient

friendly_name = "musicgen"
model = None


def generate(
    prompt: str,
    file_path: str,
    duration: int = 8,
    temperature: float = 1.0,
    cfg_coef=3,
):
    global model

    load_gpu_task(friendly_name, MusicGenClient)

    if model is None:
        model = MusicGen.get_pretrained("facebook/musicgen-small")

    model.set_generation_params(
        duration=duration, temperature=temperature, cfg_coef=cfg_coef
    )
    wav = model.generate([prompt], progress=True)

    for _, one_wav in enumerate(wav):
        audio_write(file_path, one_wav.cpu(), model.sample_rate, strategy="peak")

    del model

    return os.path.abspath(f"{file_path}.wav")


def unload():
    global model
    logging.info(f"Unloading {friendly_name}...")
    del model


def offload(for_task):
    logging.warn(f"No offload available for {friendly_name}.")
    unload()
