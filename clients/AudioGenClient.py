import logging
import os
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
from utils.gpu_utils import free_vram
from clients import AudioGenClient

friendly_name = "audiogen"
model = None

def generate(
    prompt: str,
    file_path: str,
    duration: int = 3,
    temperature: float = 1.0,
    cfg_coef=3,
):
    global model
    free_vram("audiogen", AudioGenClient)

    if model is None:
        model = AudioGen.get_pretrained("facebook/audiogen-medium")

    model.set_generation_params(
        duration=duration, temperature=temperature, cfg_coef=cfg_coef
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

    del model    

    return os.path.abspath(f"{file_path}.wav")

def unload():
    global model
    logging.info(f"Unloading {friendly_name}...")
    del model

def offload(for_task):
    logging.warn(f"No offload available for {friendly_name}.")
    unload()
