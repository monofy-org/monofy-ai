import io
import logging
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torchaudio
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
    cfg_coef: float = 3.0,
    top_p: float = 1.0,
    format: str = "wav",
    wav_bytes: bytes = None,
):
    global model
    global friendly_name

    load_gpu_task(friendly_name, MusicGenClient)

    if not model:
        model = MusicGen.get_pretrained("facebook/musicgen-small", autodetect_device())

    model.set_generation_params(
        duration=duration, temperature=temperature, cfg_coef=cfg_coef, top_p=top_p
    )

    if wav_bytes is None:
        wav = model.generate([prompt], progress=True)
    else:
        tensor, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
        wav = model.generate_continuation(tensor, sample_rate, [prompt], progress=True)

    for _, one_wav in enumerate(wav):
        audio_write(
            file_path, one_wav.cpu(), model.sample_rate, format=format, strategy="peak"
        )

    return os.path.abspath(f"{file_path}.{format}")


def unload():
    global model
    global friendly_name
    if model:
        logging.warn(f"Unloading {friendly_name}...")
        del model


def offload(for_task: str):
    global friendly_name
    logging.warn(f"No offload available for {friendly_name}.")
    unload()
