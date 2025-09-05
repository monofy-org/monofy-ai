import io
import logging
import os
import re
from typing import Literal, Optional

import numpy as np
import torch
from fastapi import Depends, HTTPException, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from scipy.io.wavfile import write

from modules.plugins import PluginBase, release_plugin, use_plugin
from settings import TTS_MODEL, TTS_VOICES_PATH, USE_DEEPSPEED
from utils.audio_utils import get_wav_bytes, wav_to_mp3
from utils.file_utils import cached_snapshot, ensure_folder_exists
from utils.gpu_utils import clear_gpu_cache
from utils.text_utils import process_text_for_tts

CHUNK_SIZE = 20

silence = np.zeros(int(0.25 * 24000), dtype=np.float32)


def get_languages():
    return {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Polish": "pl",
        "Turkish": "tr",
        "Russian": "ru",
        "Dutch": "nl",
        "Czech": "cs",
        "Arabic": "ar",
        "Chinese (Simplified)": "zh-cn",
        "Japanese": "ja",
        "Hungarian": "hu",
        "Korean": "ko",
    }


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "en"
    voice: Optional[str] = "female1"
    temperature: Optional[float] = 0.75
    speed: Optional[float] = 1.0
    pitch: Optional[float] = 1.0
    stream: Optional[bool] = False
    format: Optional[Literal["wav", "mp3"]] = "wav"


class TTSPlugin(PluginBase):
    name = "TTS"
    description = "Text-to-Speech (XTTS)"
    instance = None

    def __init__(self):
        import torch
        from TTS.tts.configs.shared_configs import BaseDatasetConfig
        from TTS.tts.configs.xtts_config import XttsAudioConfig, XttsConfig, XttsArgs

        super().__init__()

        torch.serialization.add_safe_globals(
            [XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]
        )

        ensure_folder_exists(TTS_VOICES_PATH)

        self.current_model_name = TTS_MODEL
        self.current_speaker_wav: str = None
        self.gpt_cond_latent = None
        self.prebuffer_chunks = 2
        self.busy = False
        self.interrupt = False

        self.model_path = cached_snapshot(TTS_MODEL)
        self.config = XttsConfig()
        self.config.load_json(os.path.join(self.model_path, "config.json"))

    def load_model(self):

        from TTS.tts.models.xtts import Xtts

        if self.resources.get("model"):
            model = self.resources["model"]
            if torch.cuda.is_available():
                model = model.cuda()

        else:
            model: Xtts = Xtts.init_from_config(
                self.config, device=self.device, torch_dtype=self.dtype
            )
            model.load_checkpoint(
                self.config,
                checkpoint_dir=self.model_path,
                eval=True,
                use_deepspeed=USE_DEEPSPEED,
            )
            self.current_model_name = TTS_MODEL

            if torch.cuda.is_available():
                model = model.cuda()

            self.resources["model"] = model
            self.resources["speaker_embedding"] = None
            self.resources["gpt_cond_latent"] = None

        return model

    def offload(self):
        if self.resources.get("model"):
            from TTS.tts.models.xtts import Xtts

            model: Xtts = self.resources["model"]
            model.cpu()
        if self.resources.get("speaker_embedding") is not None:
            del self.resources["speaker_embedding"]
            del self.resources["gpt_cond_latent"]
            self.current_speaker_wav = None
        clear_gpu_cache()

    def cancel(self):
        if self.busy:
            self.interrupt = True

    def load_voice(self, voice: str):
        from TTS.tts.models.xtts import Xtts

        speaker_wav = os.path.join(TTS_VOICES_PATH, f"{voice}.wav")

        if not os.path.exists(speaker_wav):
            raise HTTPException(status_code=400, detail=f"Voice {voice} not found")

        if speaker_wav != self.current_speaker_wav:
            logging.info(f"Loading voice: {voice}")
            tts: Xtts = self.resources["model"]

            gpt_cond_latent, speaker_embedding = tts.get_conditioning_latents(
                audio_path=[speaker_wav]
            )

            self.current_speaker_wav = speaker_wav
            self.resources["speaker_embedding"] = speaker_embedding
            self.resources["gpt_cond_latent"] = gpt_cond_latent

    def generate_speech(self, req: TTSRequest):
        from TTS.tts.models.xtts import Xtts

        tts: Xtts = self.load_model()

        text = process_text_for_tts(req.text)

        if not text:
            logging.info("🤷🏼 Empty text, no speech was generated.")
            return

        self.load_voice(req.voice)

        wavs = []

        segments = split_text_segments(text)

        for segment in segments:
            args: dict = {
                "text": segment,
                "language": req.language or "en",
                "speed": req.speed or 1,
                "temperature": req.temperature or 1,
                "speaker_embedding": self.resources["speaker_embedding"],
                "gpt_cond_latent": self.resources["gpt_cond_latent"],
            }
            result = tts.inference(**args)
            wavs.append(result.get("wav"))

        # Concatenate all wavs
        if wavs:
            # insert 100ms of silence between wavs
            wavs_separated = []
            for i in range(len(wavs) - 1):
                wavs_separated.append(wavs[i])
                if i < len(wavs) - 1:
                    wavs_separated.append(silence)

            wavs = wavs_separated + [wavs[-1]]

            wav = np.concatenate(wavs, axis=-1)
        else:
            wav = np.zeros(24000, dtype=np.float32)  # 1 second of silence

        return wav

    async def generate_speech_streaming(self, req: TTSRequest):
        from TTS.tts.models.xtts import Xtts

        self.busy = True
        self.interrupt = False

        self.load_model()

        tts: Xtts = self.resources["model"]

        self.load_voice(req.voice)

        chunks = []

        # split senteces by punctuation
        sentences = re.split(r"([.,!?])", req.text)
        sentences = ["".join(x).strip() for x in zip(sentences[0::2], sentences[1::2])]

        # remove empty entries
        sentences = [x for x in sentences if len(x) > 1]

        # split by sentences only when we are at 150 characters
        text_buffer = ""
        sentence_groups = []
        i = 0
        for sentence in sentences:
            text_buffer += sentence
            i += 1
            if len(text_buffer) > 80 or (len(text_buffer) > 30 and i > 2):
                if sentence_groups and len(text_buffer + sentence_groups[-1]) < 150:
                    # handle cases where we could have fit the sentence in the last group
                    sentence_groups[-1] += text_buffer
                elif len(text_buffer) > 200:
                    # handle cases where a single sentence is too long
                    sp = text_buffer.split()
                    first_half = " ".join(sp[: len(sp) // 2])
                    second_half = " ".join(sp[len(sp) // 2 :])
                    sentence_groups.append(first_half)
                    sentence_groups.append(second_half)
                else:
                    sentence_groups.append(text_buffer)
                    i = 0  # keep this rolling but clear the buffer
                text_buffer = ""  # clear this but don't reset i
        if len(text_buffer) > 0:
            if len(text_buffer) < 10:
                if sentence_groups:
                    sentence_groups[-1] += text_buffer
                else:
                    sentence_groups.append(text_buffer)
            else:
                sentence_groups.append(text_buffer)

        for speech_input in sentence_groups:
            args = dict(
                text=process_text_for_tts(speech_input),
                language=req.language,
                speed=req.speed,
                temperature=req.temperature,
                stream_chunk_size=CHUNK_SIZE,
                gpt_cond_latent=self.resources["gpt_cond_latent"],
                speaker_embedding=self.resources["speaker_embedding"],
                overlap_wav_len=512,
                # top_p=top_p,
                enable_text_splitting=False,
            )

            for chunk in tts.inference_stream(**args):
                if self.interrupt:
                    break

                self.busy = True

                chunks.append(chunk)

                if len(chunks) < self.prebuffer_chunks:
                    pass
                elif len(chunks) == self.prebuffer_chunks:
                    for chunk in chunks:
                        yield (
                            wav_to_mp3(chunk).getvalue()
                            if req.format == "mp3"
                            else chunk.cpu().numpy()
                        )
                else:
                    yield (
                        wav_to_mp3(chunk).getvalue()
                        if req.format == "mp3"
                        else chunk.cpu().numpy()
                    )

            # yield silent chunk between sentences
            yield torch.zeros(1, 11025, device="cpu").numpy()

            if self.interrupt:
                break

        if not self.interrupt:
            if len(chunks) < self.prebuffer_chunks:
                for chunk in chunks:
                    yield chunk.cpu().numpy()

        self.busy = False
        self.interrupt = False


@PluginBase.router.post(
    "/tts", response_class=StreamingResponse, tags=["Text-to-Speech (TTS)"]
)
async def tts(
    req: TTSRequest,
):
    plugin: TTSPlugin = None
    try:
        plugin = await use_plugin(TTSPlugin)
        wav = plugin.generate_speech(req)
        wave_io = io.BytesIO()
        write(wave_io, 24000, wav)
        wave_io.seek(0)

        if req.format == "mp3":
            wave_io = wav_to_mp3(wave_io, 24000)

        return StreamingResponse(wave_io, media_type=f"audio/{req.format}")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            release_plugin(plugin)


@PluginBase.router.get(
    "/tts", response_class=StreamingResponse, tags=["Text-to-Speech (TTS)"]
)
async def tts_get(
    req: TTSRequest = Depends(),
):
    return await tts(req)


@PluginBase.router.websocket("/tts/stream")
async def tts_stream(
    websocket: WebSocket,
    req: TTSRequest,
):
    plugin: TTSPlugin = None
    try:
        await websocket.accept()
        plugin = await use_plugin(TTSPlugin)
        async for chunk in plugin.generate_speech_streaming(req):
            await websocket.send_bytes(get_wav_bytes(chunk))

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            release_plugin(plugin)
        await websocket.close()


def get_voices():
    return [
        x.replace(".wav", "") for x in os.listdir(TTS_VOICES_PATH) if x.endswith(".wav")
    ]


@PluginBase.router.get("/tts/voices", tags=["Text-to-Speech (TTS)"])
async def tts_voices():

    return {"voices": get_voices()}


def split_text_segments(text: str, length=250) -> list[str]:

    segments = []

    # Split text into sentences by . ? !
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    for sentence in sentences:
        # If sentence is too long, try to split further
        parts = [sentence]
        if len(sentence) > length:
            # Try splitting by comma
            parts = re.split(r",\s*", sentence)
            parts = [p.strip() for p in parts if p.strip()]
            # If still too long, split by ' and ' or ' or '
            new_parts = []
            for part in parts:
                if len(part) > length:
                    subparts = re.split(r"\s+(and|or)\s+", part)
                    subparts = [sp.strip() for sp in subparts if sp.strip()]
                    new_parts.extend(subparts)
                else:
                    new_parts.append(part)
            parts = new_parts
            # If still too long, hard truncate near the middle at a space
            final_parts = []
            for part in parts:
                if len(part) > length:
                    # Find a space near the middle, but not the very last space
                    mid = len(part) // 2
                    space_indices = [i for i, c in enumerate(part) if c == " "]
                    split_idx = (
                        min(space_indices, key=lambda x: abs(x - mid))
                        if space_indices
                        else mid
                    )
                    first = part[:split_idx].strip()
                    second = part[split_idx:].strip()
                    if first:
                        final_parts.append(first)
                    if second:
                        final_parts.append(second)
                else:
                    final_parts.append(part)
            parts = final_parts

        for part in parts:
            if not part:
                continue
            segments.append(part)

    return segments
