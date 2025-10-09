import os
from fastapi.responses import JSONResponse
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pydantic import BaseModel
from utils.audio_utils import get_audio_from_request
from utils.file_utils import random_filename
from modules.plugins import router
from speechbrain.inference.speaker import EncoderClassifier
from sentence_transformers import util
import logging

# Set up logging to provide more helpful debug information
logging.basicConfig(level=logging.INFO)

# Determine the device to use (cuda if available, otherwise cpu)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VoiceIdRequest(BaseModel):
    audio: str
    voices: list[str] | None = None
    max_length: float | None = None

# Initialize the speaker verification model once
try:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb",
        run_opts={"device": DEVICE}
    )
except Exception as e:
    logging.error(f"Error loading SpeechBrain model: {e}. Please ensure you have the latest version and all dependencies are met.")
    raise

def create_embedding_file(wav_path, emb_path):
    """
    Loads a .wav file, extracts a speaker embedding, and saves it to a .npy file.
    Always returns a 2D tensor of shape [1, embedding_dim].
    """
    try:
        signal, fs = torchaudio.load(wav_path)
    except Exception as e:
        logging.error(f"Failed to load audio file {wav_path}: {e}")
        raise

    if fs != 16000:
        try:
            signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)
        except Exception as e:
            logging.error(f"Failed to resample audio file {wav_path}: {e}")
            raise

    # The model expects a single-channel signal with a batch dimension.
    if signal.dim() > 1:
        signal = signal[0].unsqueeze(0)
    else:
        signal = signal.unsqueeze(0)

    signal = signal.to(DEVICE)

    try:
        with torch.no_grad():
            embeddings = classifier.encode_batch(signal)
            
            # The model can return a 1D, 2D, or 3D tensor. We need to normalize this to [1, embedding_dim].
            # This is the crucial fix for the "Embedding is not 2D" error.
            if embeddings.dim() > 2:
                # If the tensor is 3D [batch, segments, dim], average along the segments.
                embeddings = torch.mean(embeddings, dim=1)
            
            if embeddings.dim() == 1:
                # If it's a 1D vector, add a batch dimension
                embeddings = embeddings.unsqueeze(0)

            # Final sanity check to ensure the tensor is 2D
            if embeddings.dim() != 2:
                 raise ValueError(f"Embedding is not 2D. Shape is {embeddings.shape}")
                
    except Exception as e:
        logging.error(f"Failed to encode batch for file {wav_path}: {e}")
        raise
    
    # Save the embedding as a 2D numpy array
    np.save(emb_path, embeddings.cpu().numpy())
    return embeddings

def voice_id(url_or_path: str, voices: list[str] = None, max_length: float | None = None):
    emb_dir = "./voices/embeddings"
    os.makedirs(emb_dir, exist_ok=True)

    input_wav_path = get_audio_from_request(url_or_path, max_length=max_length)

    if voices is None:
        voices = [
            f.split(".")[0]
            for f in os.listdir("./voices")
            if f.endswith(".wav")
        ]

    voice_embeddings = {}
    for voice in voices:
        emb_path = f"{emb_dir}/{voice}.emb.npy"
        voice_wav_path = f"./voices/{voice}.wav"

        if os.path.exists(emb_path):
            loaded_emb = np.load(emb_path)
            if loaded_emb.ndim == 1:
                loaded_emb = loaded_emb.reshape(1, -1)
            voice_embeddings[voice] = torch.from_numpy(loaded_emb).to(DEVICE)
            logging.info(f"Loaded existing embedding for '{voice}' with shape: {voice_embeddings[voice].shape}")
        else:
            embeddings = create_embedding_file(voice_wav_path, emb_path)
            voice_embeddings[voice] = embeddings
            logging.info(f"Created new embedding for '{voice}' with shape: {voice_embeddings[voice].shape}")

    input_emb_path = f"{emb_dir}/_input.emb.npy"
    input_embedding = create_embedding_file(input_wav_path, input_emb_path)
    logging.info(f"Created new embedding for input audio with shape: {input_embedding.shape}")

    os.remove(input_wav_path)
    os.remove(input_emb_path)

    best_voice = None
    best_score = -1.0
    
    # Store scores in a dictionary
    scores = {}

    for voice, voice_emb in voice_embeddings.items():
        logging.info(f"Comparing input_embedding shape: {input_embedding.shape} with voice_emb shape: {voice_emb.shape}")
        score = util.pytorch_cos_sim(input_embedding, voice_emb)
        score = score.item()
        scores[voice] = score
        logging.info(f"Similarity score for '{voice}': {score}")
        
        if score > best_score:
            best_score = score
            best_voice = voice
            
    logging.info(f"Best matching voice: '{best_voice}' with score: {best_score}")
    
    # Sort scores in descending order
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    return best_voice, sorted_scores

@router.post("/voice/id", tags=["Voice Processing"])
async def voice_id_endpoint(req: VoiceIdRequest):
    """
    Identify the closest matching voice from a set of reference voices using advanced embeddings.
    - **url_or_path**: URL or local path to the audio file to analyze.
    - **voices**: List of voice names (without extensions) to compare against. If not provided, all voices in the "./voices" directory will be used.
    Returns the name of the closest matching voice.
    """
    audio = req.audio
    voices = req.voices
    if voices is not None and not isinstance(voices, list):
        return JSONResponse(content={"error": "voices must be a list of voice names."}, status_code=400)
    if audio is None:
        return JSONResponse(content={"error": "audio is required."}, status_code=400)
    try:
        best_voice, scores = voice_id(audio, voices, req.max_length)
        return JSONResponse(content={"best_voice": best_voice, "scores": scores})
    except Exception as e:
        logging.error(f"An error occurred in voice_id_endpoint: {e}", exc_info=True)
        return JSONResponse(content={"error": "An error occurred while processing the request."}, status_code=500)