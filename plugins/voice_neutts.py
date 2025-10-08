import os
import numpy as np
import logging
from modules.plugins import PluginBase, release_plugin, use_plugin, use_plugin_unsafe
from pydantic import BaseModel
from plugins.voice_whisper import VoiceWhisperPlugin
from utils.audio_utils import load_audio, wav_to_mp3
from fastapi.responses import StreamingResponse


class VoiceNeuTTSRequest(BaseModel):
    voice: str
    text: str


class VoiceNeuTTSPlugin(PluginBase):
    name = "voice_neutts"
    description = "Text-to-Speech using NeuTTS"

    def __init__(self):
        super().__init__()
        self.resources["neutts"] = None
        self.resources["whisper"] = None

    def load_model(self):

        if self.resources["neutts"] is not None:
            return self.resources["neutts"]

        from submodules.NeuTTS.neuttsair.neutts import NeuTTSAir

        tts = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air",
            backbone_device="cuda",
            codec_repo="neuphonic/neucodec",
            codec_device="cuda",
        )

        self.resources["neutts"] = tts

        return tts

    def generate(self, text: str, voice: str, ref_text: str) -> bytes:
        from submodules.NeuTTS.neuttsair.neutts import NeuTTSAir

        model = self.load_model()

        ref_codes = model.encode_reference(f"voices/{voice}.wav")
        wav = model.infer(text, ref_codes, ref_text)

        return (24_000, wav)


async def generate_ref_text(voice: str):
    logging.info(f"Generating reference text for voice: {voice}")
    whisper: VoiceWhisperPlugin = use_plugin_unsafe(VoiceWhisperPlugin)
    
    try:
        filename = f"voices/{voice}.wav"
        # load audio from file into numpy ndarray
        sr, audio = load_audio(filename)

        voice_ndarray = np.frombuffer(audio, dtype=np.float32)
        result = await whisper.process(voice_ndarray, sr)
        print(result)
        if not result or "text" not in result:
            logging.error(f"Whisper ASR failed for voice: {voice}")
            return None
        text = result["text"].strip()
        with open(f"voices/ref/{voice}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except Exception as e:
        raise ValueError(f"Error processing transcription: {str(e)}")


@PluginBase.router.post("/voice/neutts", response_model=bytes)
async def voice_neutts(request: VoiceNeuTTSRequest):
    voice_path = f"voices/{request.voice}.wav"

    if not os.path.exists(voice_path):
        raise ValueError(f"Voice file does not exist: {voice_path}")

    if not os.path.exists("voices/ref"):
        os.mkdir("voices/ref")

    ref_path = f"voices/ref/{request.voice}.txt"

    if os.path.exists(ref_path):
        ref_text = open(ref_path, "r", encoding="utf-8").read().strip()
    else:
        ref_text = await generate_ref_text(request.voice)

    plugin: VoiceNeuTTSPlugin = None
    try:
        plugin = await use_plugin(VoiceNeuTTSPlugin)
        (sr, audio) = await plugin.generate(request.text, request.voice, ref_text)
        mp3_data = wav_to_mp3(audio, sr)

        return StreamingResponse(content=mp3_data, media_type="audio/mpeg")

    except Exception as e:
        logging.error(f"Error in TTS generation: {str(e)}", exc_info=True)
        raise ValueError(f"Error processing TTS: {str(e)}")
    finally:
        if plugin:
            release_plugin(VoiceNeuTTSPlugin)
