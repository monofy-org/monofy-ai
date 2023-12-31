import logging
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from utils.file_utils import fetch_pretrained_model


MODEL_NAME = "openai/whisper-medium"

friendly_name = "whisper"
logging.warn(f"Initializing {friendly_name}...")

current_model_name: str = MODEL_NAME
model = None
model_path = None
forced_decoder_ids = None
processor = None


def load_model(model_name: str = current_model_name):
    global current_model_name
    global model
    global model_path
    global forced_decoder_ids
    global processor

    if model is not None and model == current_model_name:
        return

    model_path = fetch_pretrained_model(model_name, "whisper")

    if model is None:
        processor = WhisperProcessor(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        current_model_name = model_name


async def process_audio_chunk(chunk: bytes):
    audio_array = np.frombuffer(chunk, dtype=np.float32)

    # Process audio chunk with the Whisper processor
    input_features = processor(
        audio_array, sampling_rate=16_000, return_tensors="pt"
    ).input_features

    # Generate token ids
    predicted_ids = model.generate(
        input_features, forced_decoder_ids=forced_decoder_ids
    )

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription


def offload(for_task: str):    
    global model
    global friendly_name
    logging.info(f"Offloading {friendly_name}...")
    model.to("cpu")


def generate():
    # TODO
    pass
