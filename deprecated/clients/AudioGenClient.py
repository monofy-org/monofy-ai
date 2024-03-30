import gc
import io
import logging
import os
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import torch
import torchaudio
from utils.gpu_utils import load_gpu_task

friendly_name = "audiogen"
logging.warn(f"Initializing {friendly_name}...")

model = None


async def generate(
    prompt: str,
    file_path: str,
    duration: int = 3,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    top_k: int = 250,
    top_p: float = 0,    
    wav_bytes: bytes = None,
):
    from clients import AudioGenClient

    global model
    global friendly_name

    async with load_gpu_task(friendly_name, AudioGenClient):

        if not model:
            model = AudioGen.get_pretrained("facebook/audiogen-medium")

        model.set_generation_params(
            duration=duration, temperature=temperature, cfg_coef=cfg_coef, top_k=top_k, top_p=top_p
        )
        if wav_bytes is None:
            wav = model.generate([prompt], progress=True)
        else:
            tensor, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
            wav = model.generate_continuation(tensor, sample_rate, [prompt], progress=True)

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
    if model is not None:
        global friendly_name
        logging.info(f"Unloading {friendly_name}...")
        del model
        model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def offload(for_task):
    # global friendly_name
    # logging.warn(f"No offload available for {friendly_name}.")
    unload()
