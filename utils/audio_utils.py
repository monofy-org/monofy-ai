import io
import numpy as np
import wave
from torch import Tensor


def get_wav_bytes(wav_tensor: Tensor):
    """Convert the NumPy array returned by inference.get("wav") to bytes with WAV header"""
    wav_bytes = _numpy_array_to_wav_bytes(wav_tensor)
    return wav_bytes


def save_wav(wav_bytes, filename: str):
    """Save the WAV bytes to a file"""
    with open(filename, "wb") as wav_file:
        wav_file.write(wav_bytes)


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
