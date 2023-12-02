import os
import logging
import base64
import time
from settings import LOG_LEVEL, TTS_MODEL, USE_DEEPSPEED
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.utils.generic_utils import get_user_data_dir
from utils.audio_utils import get_wav_bytes
from utils.text_utils import process_text_for_tts
from utils.torch_utils import autodetect_device


logging.basicConfig(level=LOG_LEVEL)

CHUNK_SIZE = 60

voices_path = os.path.join(os.path.dirname("."), "voices")
cache_path = os.path.join(os.path.dirname("."), ".cache")

default_language = "en"
default_speaker_wav = "voices/female1.wav"
default_emotion = "Neutral"
default_temperature = 0.75
default_speed = 1

# Ensure the "voices" folder exists
if not os.path.exists(voices_path):
    os.makedirs(voices_path)

# Ensure the ".cache" folder exists
if not os.path.exists(cache_path):
    os.makedirs(cache_path)


class TTSClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist
            cls._instance.load_model()
        return cls._instance

    def __init__(self):
        self.device = autodetect_device()
        logging.info(f"TTS using device: {self.device}")
        
        self.model = None
        self.model_name: str = None
        self.speaker_wav: str = None
        self.gpt_cond_latent = None
        self.speaker_embedding = None

    def load_model(self, model_name=TTS_MODEL):
        if self.model is None:
            logging.info(f"Loading model {model_name}...")
            ModelManager().download_model(model_name)
            model_path = os.path.join(
                get_user_data_dir("tts"), model_name.replace("/", "--")
            )
            config = XttsConfig()
            config.load_json(os.path.join(model_path, "config.json"))               
            config.cudnn_enable = True     
            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                config,
                checkpoint_dir=model_path,
                eval=True,
                use_deepspeed=USE_DEEPSPEED,                             
            )
            if self.device != "cpu":
                model.cuda()

            self.model = model
            self.model_name = model_name

            if self.speaker_wav is not None:
                self.load_speaker(self.speaker_wav)
            else:
                self.load_speaker(default_speaker_wav)

    def load_speaker(self, speaker_wav):
        if speaker_wav != self.speaker_wav:
            logging.info(f"Loading speaker {speaker_wav}...")
            try:
                (
                    gpt_cond_latent,
                    speaker_embedding,
                ) = self.model.get_conditioning_latents(audio_path=[speaker_wav])
            except Exception:
                logging.error(f"Couldn't get conditioning latents from {speaker_wav}")
                return

            self.speaker_wav = speaker_wav
            self.gpt_cond_latent = gpt_cond_latent
            self.speaker_embedding = speaker_embedding

    def generate_speech(
        self,
        text: str,
        speed=default_speed,
        temperature=default_temperature,
        speaker_wav=default_speaker_wav,
        language=default_language,
        emotion=default_emotion,        
    ):
        if self.model is None:
            logging.error("No model loaded")
            return None

        self.load_speaker(speaker_wav)

        result = self.model.inference(
            text=process_text_for_tts(text),
            language=language,
            gpt_cond_latent=self.gpt_cond_latent,
            speaker_embedding=self.speaker_embedding,                                 
            temperature=temperature,
            speed=speed,
            # emotion=emotion,
        )

        wav = result.get("wav")

        if wav is None:
            logging.error("Invalid WAV data.")
        else:
            return get_wav_bytes(wav)

        return wav

    def generate_speech_file(
        self,
        text: str,
        speed=default_speed,
        temperature=default_temperature,
        speaker_wav=default_speaker_wav,
        language=default_language,
        output_file: str = "output.wav",
    ):
        wav_bytes = self.generate_speech(
            text=text,
            speed=speed,
            temperature=temperature,
            speaker_wav=speaker_wav,
            language=language,
        )

        with open(output_file, "wb") as wav_file:
            wav_file.write(wav_bytes)

        logging.info(f"Saved {output_file}.")

        return output_file

    def generate_speech_base64(
        self,
        text: str,
        speed=default_speed,
        temperature=default_temperature,
        speaker_wav=default_speaker_wav,
        language=default_language,
    ):
        wav_bytes = self.generate_speech(
            text=text,
            speed=speed,
            temperature=temperature,
            speaker_wav=speaker_wav,
            language=language,
        )

        if wav_bytes is None:
            logging.error("Invalid WAV data.")
            return None

        base64_encoded = base64.b64encode(wav_bytes).decode("utf-8")
        base64_header = "data:audio/wav;base64,"

        return base64_header + base64_encoded

    def generate_speech_streaming(
        self,
        text: str,
        speed=default_speed,
        temperature=default_temperature,
        speaker_wav=default_speaker_wav,
        language=default_language,
        emotion=default_emotion,
    ):
        if self.model is None:
            logging.error("No model loaded")

        else:
            self.load_speaker(speaker_wav)
            chunks = self.model.inference_stream(
                text=process_text_for_tts(text),
                language=language,
                speed=speed,
                temperature=temperature,
                # emotion=emotion,
                stream_chunk_size=CHUNK_SIZE,
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                
                # enable_text_splitting=True,
            )

            # Send audio chunks to the client as they become available
            for i, chunk in enumerate(chunks):
                chunk = chunk.cpu().numpy()
                try:
                    yield get_wav_bytes(chunk)
                except Exception:
                    logging.info("Socket closed unexpectedly.")
                time.sleep(0.1)  # Adjust this delay as needed

    async def list_voices(self):
        try:
            voice_files = os.listdir("voices")
            voices = [voice.split(".")[0] for voice in voice_files]
            return {"voices": voices}
        except Exception as e:
            logging.error(e)
            return None
