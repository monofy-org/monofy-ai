# Assuming BasePlugin is defined elsewhere and handles resource management
# from some_framework import BasePlugin

import gc
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import Optional
from fastapi.responses import FileResponse, JSONResponse
from omegaconf import OmegaConf
from pydantic import BaseModel
from pydub import AudioSegment
from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
import torch
from modules.plugins import PluginBase, release_plugin, use_plugin, use_plugin_unsafe
from settings import CACHE_PATH
from utils.audio_utils import get_audio_from_request
from utils.file_utils import random_filename
from utils.image_utils import get_image_from_request
from utils.audio_utils import get_wav_bytes


class Img2VidMoDARequest(BaseModel):
    image: str
    audio: Optional[str] = None
    emotion: Optional[str] = "Happy"
    text: Optional[str] = None
    voice: Optional[str] = None
    seed: Optional[int] = -1
    language: Optional[str] = "en"
    speed: Optional[float] = 1.0
    temperature: Optional[float] = 0.75


class Img2VidMoDAPlugin(PluginBase):
    # We will assume a self.resources dictionary is inherited from BasePlugin
    # and a self.name attribute exists
    name = "Img2Vid (MoDA) Fast Talking Head"
    description = "Fast talking heads using MoDA"
    instance = None

    def __init__(self):
        super().__init__()
        # Configuration and path constants
        self.REPO_ID = "lixinyizju/moda"
        self.WEIGHTS_DIR = "pretrain_weights"
        self.DEFAULT_CFG_PATH = (
            "./submodules/moda/configs/audio2motion/inference/inference.yaml"
        )
        self.DEFAULT_MOTION_MEAN_STD_PATH = "./submodules/moda/src/datasets/mean.pt"
        self.DEFAULT_SILENT_AUDIO_PATH = (
            "./submodules/moda/src/examples/silent-audio.wav"
        )
        self.LIVEPORTRAIT_CONFIG = (
            "./submodules/moda/configs/audio2motion/model/liveportrait_config.yaml"
        )

        cfg = OmegaConf.load(self.DEFAULT_CFG_PATH)
        cfg.audio_model_config = (
            "./submodules/moda/configs/audio2motion/model/audio_processer_config.yaml"
        )
        cfg.motion_models_config = (
            "./submodules/moda/configs/audio2motion/model/config.yaml"
        )        
        cfg.motion_processer_config = self.LIVEPORTRAIT_CONFIG
        OmegaConf.save(cfg, self.DEFAULT_CFG_PATH)

        audio_cfg = OmegaConf.load(cfg.audio_model_config)
        audio_cfg.tmp_dir = CACHE_PATH
        # params = audio_cfg.model_params
        # params.is_chinese = False
        # audio_cfg.model_params = params
        OmegaConf.save(audio_cfg, cfg.audio_model_config)

        liveportrait_config = OmegaConf.load(self.LIVEPORTRAIT_CONFIG)
        liveportrait_config.models_config = (
            "./submodules/moda/configs/audio2motion/model/models.yaml"
        )
        liveportrait_config.crop_cfg = (
            "./submodules/moda/configs/audio2motion/model/crop_config.yaml"
        )
        liveportrait_config.lip_array = (
            "./submodules/moda/src/utils/resources/lip_array.pkl"
        )
        liveportrait_config.mask_crop = (
            "./submodules/moda/src/utils/resources/mask_template.png"
        )
        OmegaConf.save(liveportrait_config, self.LIVEPORTRAIT_CONFIG)

    def load_model(self):
        """
        Loads the MoDA pipeline, handling weight download and initialization.
        This function is designed to be idempotent.
        """
        if "pipeline" in self.resources:
            print("MoDA pipeline is already loaded.")
            return self.resources["pipeline"]

        # 1. Download weights if they don't exist
        print(f"Checking for weights at '{self.WEIGHTS_DIR}'...")
        motion_model_file = os.path.join(self.WEIGHTS_DIR, "moda", "net-200.pth")
        if not os.path.exists(motion_model_file):
            print(f"Weights not found locally. Downloading from '{self.REPO_ID}'...")
            try:
                snapshot_download(
                    repo_id=self.REPO_ID,
                    local_dir=self.WEIGHTS_DIR,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                print("Weights downloaded successfully.")
            except (
                GatedRepoError,
                RepositoryNotFoundError,
                RevisionNotFoundError,
            ) as e:
                raise Exception(f"Failed to download models from Hugging Face: {e}")
            except Exception as e:
                raise Exception(f"An unexpected error occurred during download: {e}")
        else:
            print("Found existing weights. Skipping download.")

        # 2. Initialize the pipeline
        print("Initializing MoDA pipeline...")
        try:
            # Assuming these are available from the original `src` directory
            from submodules.MoDA.src.models.inference.moda_test import (
                LiveVASAPipeline,
                emo_map,
            )

            self.emo_name_to_id = {v: k for k, v in emo_map.items()}
            pipeline = LiveVASAPipeline(
                cfg_path=self.DEFAULT_CFG_PATH,
                motion_mean_std_path=self.DEFAULT_MOTION_MEAN_STD_PATH,
            )

            print("MoDA pipeline initialized successfully.")
            self.resources["pipeline"] = pipeline
            return pipeline
        except Exception as e:
            raise Exception(f"Error initializing pipeline: {e}")

    def generate(self, source_image_path, driving_audio_path, emotion_name, cfg_scale):
        """
        Generates a talking head video using the MoDA pipeline.

        Args:
            source_image_path (str): Path to the source image.
            driving_audio_path (str): Path to the driving audio file.
            emotion_name (str): The name of the emotion.
            cfg_scale (float): The CFG scale value.

        Returns:
            str: The file path of the generated video.
        """
        pipeline = self.load_model()

        if not source_image_path or not driving_audio_path:
            raise ValueError(
                "Both source image and driving audio paths must be provided."
            )

        start_time = time.time()

        # Ensure audio is in WAV format
        wav_audio_path = self._ensure_wav_format(driving_audio_path)
        temp_wav_created = wav_audio_path != driving_audio_path

        emotion_id = self.emo_name_to_id.get(
            emotion_name, 8
        )  # Default to 'None' (ID 8)

        print(
            f"Generating video for source '{source_image_path}' and audio '{driving_audio_path}'..."
        )
        try:
            file_path = pipeline.driven_sample(
                image_path=source_image_path,
                audio_path=wav_audio_path,
                cfg_scale=float(cfg_scale),
                emo=emotion_id,
                save_dir=CACHE_PATH,
                smooth=False,
                silent_audio_path=self.DEFAULT_SILENT_AUDIO_PATH,
            )

            return os.path.join(CACHE_PATH, "final_" + os.path.basename(file_path))

        except Exception as e:
            raise Exception(f"Video generation failed: {e}")
        finally:
            # Clean up temporary WAV file if created
            if temp_wav_created and os.path.exists(wav_audio_path):
                try:
                    os.remove(wav_audio_path)
                    print(f"Cleaned up temporary WAV file: {wav_audio_path}")
                except Exception as e:
                    print(
                        f"Warning: Could not delete temporary file {wav_audio_path}: {e}"
                    )

        end_time = time.time()
        print(f"Generation completed in {end_time - start_time:.2f} seconds.")

        return result_video_path

    def _ensure_wav_format(self, audio_path):
        """
        Internal helper to convert audio to WAV format.
        """
        audio_path = Path(audio_path)
        if audio_path.suffix.lower() == ".wav":
            return str(audio_path)

        try:
            audio: AudioSegment = AudioSegment.from_file(audio_path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                wav_path = tmp_file.name
                audio.export(
                    wav_path, format="wav", parameters=["-ar", "16000", "-ac", "1"]
                )
            return wav_path
        except Exception as e:
            raise Exception(f"Failed to convert audio to WAV: {e}")


# Example of how the class would be used
# plugin = MoDAPlugin()
# plugin.resources = {} # Simulating the inherited resource dictionary
# try:
#     # The generate function will automatically call load_model
#     video_path = plugin.generate(
#         source_image_path="path/to/image.jpg",
#         driving_audio_path="path/to/audio.mp3",
#         emotion_name="Happy",
#         cfg_scale=1.5
#     )
#     print(f"Generated video saved at: {video_path}")
# except Exception as e:
#     print(f"An error occurred: {e}")


@PluginBase.router.post("/img2vid/moda")
async def img2vid_moda(req: Img2VidMoDARequest):
    try:
        image = get_image_from_request(req.image, return_path=True)
    except Exception:
        return JSONResponse(
            {"error": "Failed to fetch image from supplied URL"}, status_code=415
        )

    if req.audio:
        audio = get_audio_from_request(req.audio)
    elif req.text and req.voice:
        from plugins.tts import TTSPlugin, TTSRequest

        tts: TTSPlugin = use_plugin_unsafe(TTSPlugin)
        audio_bytes = tts.generate_speech(
            TTSRequest(
                text=req.text,
                voice=req.voice,
                language=req.language,
                speed=req.speed,
                temperature=req.temperature,
            )
        )
        audio = random_filename("wav")
        with open(audio, "wb") as f:
            f.write(get_wav_bytes(audio_bytes))
    else:
        return JSONResponse({"error": "Bad request"}, status_code=400)

    plugin: Img2VidMoDAPlugin = None

    try:
        plugin = await use_plugin(Img2VidMoDAPlugin)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        result = plugin.generate(
            source_image_path=image,
            driving_audio_path=audio,
            emotion_name=req.emotion,
            cfg_scale=1.5,
        )

        return FileResponse(
            result, media_type="video/mp4", filename=os.path.basename(result)
        )
    except Exception as e:
        logging.error(e, exc_info=True)
        return JSONResponse(
            content={"error": "Error generating video"}, status_code=500
        )
    finally:
        if plugin:
            release_plugin(Img2VidMoDAPlugin)
