import base64
import io
import logging
import os
import numpy as np
import wave
import torch
import torchaudio
import soundfile as sf
import librosa
from utils.file_utils import download_to_cache
from utils.video_utils import get_video_from_request


def resample(wav: np.ndarray, original_sr: int, target_sr: int):
    wav: np.ndarray = librosa.resample(wav, orig_sr=original_sr, target_sr=target_sr)
    return wav


def resample_wav(wav: bytes, target_sr: int):
    wav, sr = sf.read(io.BytesIO(wav))
    wav: np.ndarray = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.tobytes()


def get_wav_bytes(wav_tensor: torch.Tensor):
    """Convert the NumPy array returned by inference.get("wav") to bytes with WAV header"""
    wav_bytes = _numpy_array_to_wav_bytes(wav_tensor)
    return wav_bytes


def wav_to_mp3(wav: io.BytesIO | torch.Tensor | np.ndarray, sample_rate=24000):
    if isinstance(wav, io.BytesIO):
        wav.seek(0)
        wav = torch.frombuffer(wav.getbuffer(), dtype=torch.int16)

    if isinstance(wav, torch.Tensor):
        data = wav.unsqueeze(0).cpu()
        mp3_io = io.BytesIO()
        torchaudio.save(mp3_io, data, sample_rate, format="mp3")
        mp3_io.seek(0)
        return mp3_io
    else:
        mp3_io = io.BytesIO()
        sf.write(mp3_io, wav, sample_rate, format="mp3")
        mp3_io.seek(0)
        return mp3_io


def save_wav(wav_bytes, filename: str):
    with open(filename, "wb") as wav_file:
        wav_file.write(wav_bytes)


def audio_to_base64(wav_bytes: bytes):
    return base64.b64encode(wav_bytes).decode("utf-8")


def get_audio_loop(y: np.ndarray, sr: int):
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, aggregate=np.median, center=False
    )

    # Compute a log-power Mel spectrogram focusing on frequencies below 100Hz
    S = librosa.feature.melspectrogram(y=y, sr=sr, fmax=100)
    log_S = librosa.amplitude_to_db(S, ref=np.max)

    # Combine onset strength with the spectrogram
    combined = onset_env * log_S

    # Find the onset point with the highest combined energy within the first few seconds of the audio
    window_size = int(sr * 2)  # Adjust this window size as needed
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples", hop_length=256)

    if len(onsets) > 0:
        # If onsets are detected, use the first onset
        start_frame = onsets[0]
    else:
        # If no onsets are detected, find the point with the highest combined energy
        start_frame = np.argmax(combined[:window_size])

    # Ensure start_frame is within the audio bounds
    start_frame = min(start_frame, len(y) - 1)
    # Estimate tempo (BPM) and beats
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=256
    )

    # Calculate the duration of one beat in seconds
    beat_duration = 60 / tempo

    # Extract a loop of 8 beats if possible, otherwise 4 beats
    loop_duration = (
        beat_duration * 8 if len(y) >= beat_duration * 8 * sr else beat_duration * 4
    )
    loop_samples = int(loop_duration * sr)

    # Extract the loop starting from the selected beat
    loop = y[start_frame : start_frame + loop_samples]

    logging.info(f"Loop duration: {loop_duration} seconds")

    loop_io = io.BytesIO()
    sf.write(loop_io, loop, sr, format="wav")
    loop_io.seek(0)
    return loop_io


def _numpy_array_to_wav_bytes(numpy_array, channels=1, sample_rate=24000):
    """Create a BytesIO object to store the WAV file"""
    wav_bytes_io = io.BytesIO()

    wav_int16 = (numpy_array * 32767).astype(np.int16)

    # Create a wave file with a single channel (mono)
    with wave.open(wav_bytes_io, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(wav_int16.tobytes())

    # Get the bytes from the BytesIO object
    wav_bytes = wav_bytes_io.getvalue()

    return wav_bytes


def get_audio_from_request(url_or_path: str):
    logging.info(f"Downloading audio from {url_or_path}...")

    ext = url_or_path.split(".")[-1]

    if ext in ["mp3", "wav"]:
        if os.path.exists(url_or_path):
            return url_or_path
        else:
            return download_to_cache(url_or_path, ext)

    else:
        return get_video_from_request(url_or_path, audio_only=True)
