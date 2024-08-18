import base64
import io
import logging
import os
from fastapi.responses import FileResponse
import numpy as np
import wave
import requests
from torch import Tensor
import soundfile as sf
import librosa

from plugins.extras.youtube import YouTubeDownloadRequest, download


def resample(wav: np.ndarray, original_sr: int, target_sr: int):
    wav: np.ndarray = librosa.resample(wav, orig_sr=original_sr, target_sr=target_sr)
    return wav


def resample_wav(wav: bytes, target_sr: int):
    wav, sr = sf.read(io.BytesIO(wav))
    wav: np.ndarray = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.tobytes()


def get_wav_bytes(wav_tensor: Tensor):
    """Convert the NumPy array returned by inference.get("wav") to bytes with WAV header"""
    wav_bytes = _numpy_array_to_wav_bytes(wav_tensor)
    return wav_bytes


def wav_to_mp3(wav, sample_rate=24000):
    import torchaudio

    data = wav.unsqueeze(0).cpu()
    # check for empty
    if data.size(1) > 0:
        mp3_chunk = io.BytesIO()
        torchaudio.save(mp3_chunk, data, sample_rate, format="mp3")
        mp3_chunk.seek(0)
        return mp3_chunk.getvalue()


def wav_io(wav_bytes: bytes, sampling_rate: int, format: str = "wav"):
    b = io.BytesIO()
    sf.write(b, wav_bytes, sampling_rate, format="wav")
    b.seek(0)
    return b


def save_wav(wav_bytes, filename: str):
    """Save the WAV bytes to a file"""
    with open(filename, "wb") as wav_file:
        wav_file.write(wav_bytes)


def audio_to_base64(wav_bytes: bytes):
    return base64.b64encode(wav_bytes).decode("utf-8")


def get_audio_loop(wav_io: io.BytesIO):
    # Load audio file
    y, sr = librosa.load(wav_io)

    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, center=False)

    # Compute a log-power Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    log_S = librosa.amplitude_to_db(S, ref=np.max)

    # Combine onset strength with the spectrogram
    combined = onset_env * log_S

    # Find the onset point with the highest combined energy within the first few seconds of the audio
    window_size = int(sr * 2)  # Adjust this window size as needed
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples")
    start_frame = (
        np.argmax(combined[:window_size])
        if len(onsets) == 0
        else min(onsets, key=lambda x: abs(x - 0))
    )

    # Estimate tempo (BPM) for later use
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, start_bpm=100)

    # Calculate the duration of one beat in seconds
    beat_duration = 60 / tempo

    # Extract a loop of 8 beats if possible, otherwise 4 beats
    loop_duration = (
        beat_duration * 8 if len(y) >= beat_duration * 8 * sr else beat_duration * 4
    )
    loop_samples = int(loop_duration * sr)

    # Extract the loop starting from the selected beat
    loop = y[start_frame : start_frame + loop_samples]

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


def get_audio_from_request(url_or_path: str, save_path: str):

    logging.info(f"Downloading audio from {url_or_path}...")

    ext = url_or_path.split(".")[-1]

    if ext in ["mp3", "wav"]:
        if os.path.exists(url_or_path):
            return url_or_path
        else:
            response = requests.get(url_or_path)
            with open(save_path, "wb") as f:
                f.write(response.content)
            return save_path

    elif "youtube.com" in url_or_path or "youtu.be" in url_or_path:
        response: FileResponse = download(
            YouTubeDownloadRequest(url=url_or_path, audio_only=True)
        )
        logging.info(f"Downloaded {response.path}")
        return response.path
    else:
        raise ValueError(f"Unsupported audio format: {ext}")
