import io
import os
import logging
import re
from typing import Literal, Optional
from fastapi.responses import StreamingResponse
from scipy.io.wavfile import write
import torch
import torchaudio
from settings import TTS_MODEL, TTS_VOICES_PATH, USE_DEEPSPEED
from utils.audio_utils import get_wav_bytes
from utils.file_utils import ensure_folder_exists, cached_snapshot
from utils.text_utils import process_text_for_tts
from fastapi import Depends, HTTPException, WebSocket
from modules.plugins import PluginBase, use_plugin
from pydantic import BaseModel


CHUNK_SIZE = 20


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
        from submodules.TTS.TTS.tts.configs.xtts_config import XttsConfig
        from submodules.TTS.TTS.tts.models.xtts import Xtts

        super().__init__()

        ensure_folder_exists(TTS_VOICES_PATH)

        self.current_model_name = TTS_MODEL
        self.current_speaker_wav: str = None
        self.gpt_cond_latent = None
        self.prebuffer_chunks = 2
        self.busy = False
        self.interrupt = False

        model_path = cached_snapshot(TTS_MODEL)

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))

        model = Xtts.init_from_config(
            config, device=self.device, torch_dtype=self.dtype
        )
        model.load_checkpoint(
            config,
            checkpoint_dir=model_path,
            eval=True,
            use_deepspeed=USE_DEEPSPEED,
        )
        self.current_model_name = TTS_MODEL

        if torch.cuda.is_available():
            model = model.cuda()

        self.resources["model"] = model
        self.resources["config"] = config
        self.resources["speaker_embedding"] = None
        self.resources["gpt_cond_latent"] = None

    def cancel(self):
        if self.busy:
            self.interrupt = True

    def load_voice(self, voice: str):
        from submodules.TTS.TTS.tts.models.xtts import Xtts

        speaker_wav = os.path.join(TTS_VOICES_PATH, f"{voice}.wav")

        if speaker_wav != self.current_speaker_wav:
            tts: Xtts = self.resources["model"]
            (
                gpt_cond_latent,
                speaker_embedding,
            ) = tts.get_conditioning_latents(audio_path=[speaker_wav])

            self.current_speaker_wav = speaker_wav
            self.resources["speaker_embedding"] = speaker_embedding
            self.resources["gpt_cond_latent"] = gpt_cond_latent

    def generate_speech(self, req: TTSRequest):

        from submodules.TTS.TTS.tts.models.xtts import Xtts

        tts: Xtts = self.resources["model"]

        text = process_text_for_tts(req.text)

        self.load_voice(req.voice)

        args = dict(
            {
                "text": text,
                "language": req.language,
                "speed": req.speed,
                "temperature": req.temperature,
                "speaker_embedding": self.resources["speaker_embedding"],
                "gpt_cond_latent": self.resources["gpt_cond_latent"],
            }
        )

        result = tts.inference(
            **args,
        )

        wav = result.get("wav")
        return wav

    async def generate_speech_streaming(self, req: TTSRequest):

        from submodules.TTS.TTS.tts.models.xtts import Xtts

        self.busy = True
        self.interrupt = False

        tts: Xtts = self.resources["model"]

        self.load_voice(req.voice)

        gpt_cond_latent = self.resources["gpt_cond_latent"]
        speaker_embedding = self.resources["speaker_embedding"]
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
            if len(text_buffer) > 100 or i > 2:
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

            for chunk in tts.inference_stream(
                text=process_text_for_tts(speech_input),
                language=req.language,
                speed=req.speed,
                temperature=req.temperature,
                stream_chunk_size=CHUNK_SIZE,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                overlap_wav_len=512,
                # top_p=top_p,
                enable_text_splitting=False,
            ):
                if self.interrupt:
                    break

                chunks.append(chunk)

                def mp3(chunk):
                    data = chunk.unsqueeze(0).cpu()
                    # check for empty
                    if data.size(1) > 0:
                        mp3_chunk = io.BytesIO()
                        torchaudio.save(mp3_chunk, data, 24000, format="mp3")
                        mp3_chunk.seek(0)
                        return mp3_chunk.getvalue()

                if len(chunks) < self.prebuffer_chunks:
                    pass
                elif len(chunks) == self.prebuffer_chunks:
                    for chunk in chunks:
                        yield mp3(chunk) if req.format == "mp3" else chunk.cpu().numpy()
                else:
                    yield mp3(chunk) if req.format == "mp3" else chunk.cpu().numpy()

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
    try:
        plugin: TTSPlugin = await use_plugin(TTSPlugin, True)
        wav = plugin.generate_speech(req)
        wav_output = io.BytesIO()
        write(wav_output, 24000, wav)
        wav_output.seek(0)
        return StreamingResponse(wav_output, media_type="audio/wav")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
    await websocket.accept()
    try:
        plugin: TTSPlugin = await use_plugin(TTSPlugin, True)
        async for chunk in plugin.generate_speech_streaming(req):
            await websocket.send_bytes(get_wav_bytes(chunk))

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await websocket.close()


@PluginBase.router.get("/tts/voices", tags=["Text-to-Speech (TTS)"])
async def tts_voices():
    voices = [
        x.replace(".wav", "") for x in os.listdir(TTS_VOICES_PATH) if x.endswith(".wav")
    ]
    return {"voices": voices}
