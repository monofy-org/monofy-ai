import io
import os
import logging
from fastapi.responses import StreamingResponse
from scipy.io.wavfile import write
from settings import TTS_MODEL, TTS_VOICES_PATH, USE_DEEPSPEED
from submodules.TTS.TTS.utils.generic_utils import get_user_data_dir
from submodules.TTS.TTS.utils.manage import ModelManager
from utils.audio_utils import get_wav_bytes
from utils.file_utils import ensure_folder_exists
from utils.text_utils import process_text_for_tts
from utils.gpu_utils import autodetect_dtype, autodetect_device
from fastapi import Depends, HTTPException, WebSocket
from modules.plugins import PluginBase, use_plugin
from pydantic import BaseModel


CHUNK_SIZE = 60


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    voice: str = "female1"
    temperature: float = 0.75
    speed: int = 1


class TTSPlugin(PluginBase):

    name = "TTS"
    description = "Text-to-Speech (XTTS)"
    instance = None
    plugins = ["VoiceWhisperPlugin", "ExllamaV2Plugin"]

    def __init__(self):
        import torch
        from submodules.TTS.TTS.demos.xtts_ft_demo.xtts_demo import XTTS_MODEL
        from submodules.TTS.TTS.tts.configs.xtts_config import XttsConfig
        from submodules.TTS.TTS.tts.models.xtts import Xtts

        super().__init__()

        ensure_folder_exists(TTS_VOICES_PATH)

        self.current_model_name = TTS_MODEL
        self.current_speaker_wav: str = None
        self.gpt_cond_latent = None

        #model_name = "coqui/XTTS-v2"

        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        ModelManager().download_model(model_name)
        model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
        
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
        self.current_model_name = XTTS_MODEL

        if torch.cuda.is_available():
            model.cuda()

        self.resources["model"] = model
        self.resources["config"] = config
        self.resources["speaker_embedding"] = None
        self.resources["gpt_cond_latent"] = None

    def load_voice(self, voice: str):
        from submodules.TTS.TTS.tts.models.xtts import Xtts

        speaker_wav = os.path.join(TTS_VOICES_PATH, f"{voice}.wav")

        if speaker_wav != self.current_speaker_wav:
            tts: Xtts = self.resources["model"]
            (
                gpt_cond_latent,
                speaker_embedding,
            ) = tts.get_conditioning_latents(audio_path=[speaker_wav])

            gpt_cond_latent.to(
                autodetect_device(), dtype=autodetect_dtype(), non_blocking=True
            )

            self.current_speaker_wav = speaker_wav
            self.resources["speaker_embedding"] = speaker_embedding
            self.resources["gpt_cond_latent"] = gpt_cond_latent
        else:
            speaker_embedding = self.resources["speaker_embedding"]
            gpt_cond_latent = self.resources["gpt_cond_latent"]

    async def generate_speech(self, req: TTSRequest):

        from submodules.TTS.TTS.tts.models.xtts import Xtts

        tts: Xtts = self.resources["model"]

        text = process_text_for_tts(req.text)

        self.load_voice(req.voice)

        speaker_embedding = self.resources["speaker_embedding"]
        gpt_cond_latent = self.resources["gpt_cond_latent"]

        args = req.__dict__.copy()
        args.pop("text")
        args.pop("voice")

        result = tts.inference(
            text=text,
            speaker_embedding=speaker_embedding,
            gpt_cond_latent=gpt_cond_latent,
            **args,
        )

        wav = result.get("wav")
        return wav

    async def generate_speech_streaming(self, req: TTSRequest):

        from submodules.TTS.TTS.tts.models.xtts import Xtts

        tts: Xtts = self.resources["model"]

        self.load_voice(req.voice)

        gpt_cond_latent = self.resources["gpt_cond_latent"]
        speaker_embedding = self.resources["speaker_embedding"]

        for chunk in tts.inference_stream(
            text=process_text_for_tts(req.text),
            language=req.language,
            speed=req.speed,
            temperature=req.temperature,
            stream_chunk_size=CHUNK_SIZE,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            # top_p=top_p,
            # enable_text_splitting=True,
        ):
            # if format == "mp3":
            #    #encode chunk as mp3 file
            #    mp3_chunk = io.BytesIO()
            #    torchaudio.save(mp3_chunk, chunk.unsqueeze(0), 24000, format="mp3")
            #    yield np.array(mp3_chunk)
            # else:
            yield chunk.cpu().numpy()


@PluginBase.router.post("/tts", response_class=StreamingResponse, tags=["Text-to-Speech (TTS)"])
async def tts(
    req: TTSRequest,
):
    try:
        plugin: TTSPlugin = await use_plugin(TTSPlugin, True)
        wav = await plugin.generate_speech(req)
        wav_output = io.BytesIO()
        write(wav_output, 24000, wav)
        wav_output.seek(0)
        return StreamingResponse(wav_output, media_type="audio/wav")
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@PluginBase.router.get("/tts", response_class=StreamingResponse, tags=["Text-to-Speech (TTS)"])
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
