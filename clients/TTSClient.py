import gc
import io
import os
import logging
import time
import torch
from scipy.io.wavfile import write
from settings import TTS_MODEL, TTS_VOICES_PATH, USE_DEEPSPEED
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from utils import gpu_utils
from utils.file_utils import ensure_folder_exists, fetch_pretrained_model
from utils.misc_utils import print_completion_time
from utils.text_utils import process_text_for_tts
from utils.gpu_utils import autodetect_dtype, load_gpu_task, autodetect_device
from clients import TTSClient


friendly_name = "tts"
logging.warn(f"Initializing {friendly_name}...")

ensure_folder_exists(TTS_VOICES_PATH)

CHUNK_SIZE = 60

default_language = "en"
default_speaker_wav = os.path.join(TTS_VOICES_PATH, "female1.wav")
default_emotion = "Neutral"
default_temperature = 0.75
default_speed = 1

current_model_name = TTS_MODEL
model = None
current_speaker_wav: str = None
gpt_cond_latent = None
speaker_embedding = None


def load_model(model_name=current_model_name):
    global current_model_name
    global model    

    if model is None or model_name != current_model_name:
        model_path = fetch_pretrained_model(model_name)
        logging.warn("Loading model: " + model_name)
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        config.cudnn_enable = torch.backends.cudnn.is_available()
        model = Xtts.init_from_config(
            config, device=autodetect_device(), torch_dtype=autodetect_dtype()
        )        
        model.load_checkpoint(
            config,
            checkpoint_dir=model_path,
            eval=True,
            use_deepspeed=USE_DEEPSPEED,
        )
        current_model_name = model_name
    
    if torch.cuda.is_available():
        model.cuda()


def offload(for_task: str):
    global model
    global friendly_name
    if model.device != "cpu":
        logging.info(f"Offloading {friendly_name}...")
        model.to("cpu")


def load_speaker(speaker_wav=default_speaker_wav):
    global model
    global gpt_cond_latent
    global speaker_embedding
    global current_speaker_wav

    if model is None:
        load_model()
    if model is None:
        logging.error(f"{friendly_name} failed to load model.")
        return

    if speaker_wav != current_speaker_wav:
        logging.warn(f"Loading speaker {speaker_wav}...")
        try:
            (
                gpt_cond_latent,
                speaker_embedding,
            ) = model.get_conditioning_latents(audio_path=[speaker_wav])
        except Exception:
            logging.error(f"Couldn't get conditioning latents from {speaker_wav}")
            return

        current_speaker_wav = speaker_wav


async def generate_speech(
    text: str,
    speed=default_speed,
    temperature=default_temperature,
    speaker_wav=default_speaker_wav,
    language=default_language,
    emotion=default_emotion,
) -> bytes:
    global model
    global gpt_cond_latent
    global speaker_embedding
    async with load_gpu_task("tts", TTSClient):

        load_speaker(speaker_wav)

        result = model.inference(
            text=process_text_for_tts(text),
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            speed=speed,
            # emotion=emotion,
        )

        wav = result.get("wav")
        wav_output = io.BytesIO()
        write(wav_output, 24000, wav)
        wav_bytes: bytes = wav_output.getvalue()

        return wav_bytes


def generate_speech_file(
    text: str,
    speed=default_speed,
    temperature=default_temperature,
    speaker_wav=default_speaker_wav,
    language=default_language,
    output_file: str = "output.wav",
):
    with gpu_utils.gpu_thread_lock:

        load_gpu_task("tts", TTSClient)

        load_speaker(speaker_wav)

        with torch.no_grad():
            wav_bytes = generate_speech(
                text=process_text_for_tts(text),
                speed=speed,
                temperature=temperature,
                speaker_wav=speaker_wav,
                language=language,
            )

    with open(output_file, "wb") as wav_file:
        wav_file.write(wav_bytes)

    logging.info(f"Saved {output_file}.")

    return output_file


def generate_speech_streaming(
    text: str,
    speed=default_speed,
    temperature=default_temperature,
    top_p: float = 0.85,
    speaker_wav=default_speaker_wav,
    language=default_language,
    emotion=default_emotion,
    format: str = "numpy",
):
        load_gpu_task("tts", TTSClient)

        global model

        if model is None:
            load_model()

        load_speaker(speaker_wav)

        start_time = time.time()

        for chunk in model.inference_stream(
            text=process_text_for_tts(text),
            language=language,
            speed=speed,
            temperature=temperature,
            # emotion=emotion,
            stream_chunk_size=CHUNK_SIZE,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            top_p=top_p,
            # enable_text_splitting=True,
        ):
            # if format == "mp3":
            #    #encode chunk as mp3 file
            #    mp3_chunk = io.BytesIO()
            #    torchaudio.save(mp3_chunk, chunk.unsqueeze(0), 24000, format="mp3")
            #    yield np.array(mp3_chunk)
            # else:
            yield chunk.cpu().numpy()

        print_completion_time(start_time, "TTS streaming")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def list_voices():
    global voices
    try:
        voice_files = os.listdir(TTS_VOICES_PATH)
        voices = [voice.split(".")[0] for voice in voice_files]
        return {"voices": voices}
    except Exception as e:
        logging.error(e)
        return None
