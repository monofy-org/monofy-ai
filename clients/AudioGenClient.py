import logging
import os
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
from utils.gpu_utils import load_gpu_task
from clients import AudioGenClient

friendly_name = "audiogen"
logging.warn(f"Initializing {friendly_name}...")

model = None


def generate(
    prompt: str,
    file_path: str,
    duration: int = 3,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    top_p: float = 1.0,
):
    global model
    global friendly_name

    load_gpu_task(friendly_name, AudioGenClient)

    if not model:
        model = AudioGen.get_pretrained("facebook/audiogen-medium")

    model.set_generation_params(
        duration=duration, temperature=temperature, cfg_coef=cfg_coef, top_p=top_p
    )
    wav = model.generate([prompt], progress=True)

    for _, one_wav in enumerate(wav):
        audio_write(
            file_path,
            one_wav.cpu(),
            model.sample_rate,
            strategy="peak",
            loudness_compressor=True,
        )

    return os.path.abspath(f"{file_path}.wav")


def unload():
    global model
    global friendly_name
    logging.info(f"Unloading {friendly_name}...")
    del model


def offload(for_task):
    global friendly_name
    logging.warn(f"No offload available for {friendly_name}.")
    unload()
