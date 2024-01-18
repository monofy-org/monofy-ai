import logging
import io
import os
import whisper
import torch
import torchaudio
from utils.gpu_utils import autodetect_device, is_fp16_available

# from utils.audio_utils import resample_wav

MODEL_NAME = "openai/whisper-medium"

friendly_name = "whisper"
logging.warn(f"Initializing {friendly_name}...")

model = None
device = autodetect_device()


def load_model():
    global model

    model = whisper.load_model(
        "base", device=device, download_root=os.path.join("models", "whisper")
    )
    model.eval()


def transcribe(wav_bytes):
    global model

    if model is None:
        load_model()

    # Load wav bytes into a Tensor
    wav_tensor, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
    wav_tensor.to(device, dtype=torch.float16 if is_fp16_available else torch.float32)
    return model.transcribe(wav_tensor[0], word_timestamps=True)


def offload(for_task: str):
    global model
    global friendly_name
    logging.info(f"Offloading {friendly_name}...")
    model.to("cpu")
