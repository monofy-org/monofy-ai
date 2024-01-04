import logging
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from utils.gpu_utils import load_gpu_task
from clients import MusicGenClient
from utils.gpu_utils import autodetect_device


friendly_name = "musicgen"
logging.warn(f"Initializing {friendly_name}...")
model = None


def generate(
    prompt: str,
    file_path: str,
    duration: int = 8,
    temperature: float = 1.0,
    cfg_coef=3,
):
    global model
    global friendly_name

    load_gpu_task(friendly_name, MusicGenClient)

    if model is None:
        model = MusicGen.get_pretrained("facebook/musicgen-small", autodetect_device())

    model.set_generation_params(
        duration=duration, temperature=temperature, cfg_coef=cfg_coef
    )
    wav = model.generate([prompt], progress=True)

    for _, one_wav in enumerate(wav):
        audio_write(file_path, one_wav.cpu(), model.sample_rate, strategy="peak")

    return os.path.abspath(f"{file_path}.wav")


def unload():
    global model
    global friendly_name
    logging.warn(f"Unloading {friendly_name}...")
    model = None


def offload(for_task: str):
    global friendly_name
    logging.warn(f"No offload available for {friendly_name}.")
    unload()
