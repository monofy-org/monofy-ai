import gc
import io
import logging
import os
import time
from audiocraft.data.audio import audio_write
import torch
import torchaudio
from utils.gpu_utils import autodetect_device, load_gpu_task, set_seed
from clients import MusicGenClient
from utils.file_utils import import_model
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from utils.misc_utils import print_completion_time
from settings import MUSICGEN_MODEL


friendly_name = "musicgen"
logging.warn(f"Initializing {friendly_name}...")
processor: AutoProcessor = None
model: MusicgenForConditionalGeneration = None
sampling_rate = None

def generate(
    prompt: str,
    output_path: str,
    duration: int = 8,
    temperature: float = 1.0,
    guidance_scale: float = 3.0,
    top_p: float = 0.9,
    format: str = "wav",
    wav_bytes: bytes = None,
    seed: int = -1,
):
    global model
    global processor
    global friendly_name
    global sampling_rate

    load_gpu_task(friendly_name, MusicGenClient)

    if not model:
        processor = import_model(
            AutoProcessor,
            MUSICGEN_MODEL,
            device=autodetect_device(),
            allow_fp16=False,
            allow_bf16=False,
            set_variant_fp16=False,
        )
        model = import_model(
            MusicgenForConditionalGeneration,
            MUSICGEN_MODEL,
            allow_fp16=False,
            allow_bf16=False,
            set_variant_fp16=False,
        )

        sampling_rate = model.config.audio_encoder.sampling_rate

    # model.set_generation_params(
    #    duration=duration, temperature=temperature, cfg_coef=cfg_coef, top_p=top_p
    # )

    start_time = time.time()

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt", 
        sampling_rate=sampling_rate,      
    ).to(autodetect_device())

    logging.info(f"Generating {duration}s of music...")

    set_seed(seed)

    if wav_bytes is None:
        wav = model.generate(
            **inputs,
            max_new_tokens=int(duration * 50),
            temperature=temperature,
            top_p=top_p,
            guidance_scale=guidance_scale,
        )
    else:
        tensor, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
        wav = model.generate_continuation(tensor, sample_rate, [prompt], progress=True)

    for _, one_wav in enumerate(wav):
        audio_write(output_path, one_wav.cpu(), sampling_rate, format=format, strategy="peak")

    print_completion_time(start_time, "musicgen")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()        

    return os.path.abspath(f"{output_path}.{format}")


def unload():
    global model
    global processor

    if model is not None:
        global friendly_name
        if model:
            logging.warn(f"Unloading {friendly_name}...")
            del model
            del processor
            model = None
            processor = None


def offload(for_task: str):
    global friendly_name
    # logging.warn(f"No offload available for {friendly_name}.")
    unload()
