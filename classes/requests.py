# These requests are shared between various plugins.
# They are separated to prevent unnecessary or circular imports.

from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional

from settings import (
    IMG2VID_MAX_FRAMES,
    SD_DEFAULT_LORA_STRENGTH,
    SD_DEFAULT_MODEL_INDEX,
    TXT2VID_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_UPSCALE_STRENGTH,
    SD_USE_FREEU,
    SDXL_USE_REFINER,
    TXT2VID_DEFAULT_MODEL_INDEX,
)


class ImageProcessingRequest(BaseModel):
    image: str
    return_json: Optional[bool] = False


class Txt2ImgRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str = ""
    negative_prompt: Optional[str] = ""
    width: Optional[int] = None
    height: Optional[int] = None
    guidance_scale: Optional[float] = TXT2VID_DEFAULT_GUIDANCE_SCALE
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = -1
    model_index: Optional[int] = SD_DEFAULT_MODEL_INDEX
    scheduler: Optional[str] = None
    nsfw: Optional[bool] = False
    face_prompt: Optional[str] = None
    upscale: Optional[float] = 0
    strength: Optional[float] = SD_DEFAULT_UPSCALE_STRENGTH
    lora_strength: Optional[float] = SD_DEFAULT_LORA_STRENGTH
    auto_lora: Optional[bool] = True
    freeu: Optional[bool] = SD_USE_FREEU
    hi: Optional[bool] = False
    hyper: Optional[bool] = False
    return_json: Optional[bool] = False
    image: Optional[str] = None
    image2: Optional[str] = None
    invert: Optional[bool] = False
    adapter: Optional[Literal["canny", "depth", "qr"]] = None
    tiling: Optional[bool] = False
    controlnet: Optional[str] = None
    use_refiner: Optional[bool] = SDXL_USE_REFINER


class Txt2VidRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prompt: str = ""
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 576
    height: Optional[int] = 576
    guidance_scale: Optional[float] = TXT2VID_DEFAULT_GUIDANCE_SCALE
    lora_strength: Optional[float] = 0.8
    num_frames: Optional[int] = 16
    num_inference_steps: Optional[int] = 8
    nsfw: Optional[bool] = False
    hi: Optional[bool] = False
    auto_lora: Optional[bool] = True
    fps: Optional[float] = 6
    seed: Optional[int] = -1
    image: Optional[str] = None
    interpolate_film: Optional[int] = 0
    interpolate_rife: Optional[int] = 0
    fast_interpolate: Optional[bool] = False
    audio: Optional[str] = None
    mmaudio_prompt: Optional[str] = ""  # prompt for MMAudioPlugin
    mmaudio_negative_prompt: Optional[str] = ""  # prompt for MMAudioPlugin
    model_index: Optional[int] = TXT2VID_DEFAULT_MODEL_INDEX
    clip_index: Optional[int] = None
    motion_adapter: Optional[Literal["animatediff", "animatelcm"]] = "animatediff"
    scheduler: Optional[Literal["euler_a", "lcm", "sde", "tcd", "custom"]] = "lcm"
    use_animatelcm: Optional[bool] = False
    use_lightning: Optional[bool] = False

class Img2VidRequest(BaseModel):
    image: str
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 576
    height: Optional[int] = 576
    num_inference_steps: Optional[int] = 20
    guidance_scale: Optional[float] = 9.0
    seed: Optional[int] = -1
    fps: Optional[int] = 8
    interpolate_film: Optional[int] = 0
    interpolate_rife: Optional[int] = 0
    num_frames: Optional[int] = IMG2VID_MAX_FRAMES
    fast_interpolate: Optional[int] = False
    audio: Optional[str] = None
    mmaudio_prompt: Optional[str] = ""  # prompt for MMAudioPlugin
    mmaudio_negative_prompt: Optional[str] = ""  # prompt for MMAudioPlugin

class ModelInfoRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_index: Optional[int] = None


class MusicGenRequest(BaseModel):
    prompt: str
    duration: Optional[int] = 10
    temperature: Optional[float] = 1.0
    guidance_scale: Optional[float] = 7
    num_samples: Optional[int] = 500
    format: Optional[str] = "wav"
    seed: Optional[int] = -1
    top_p: Optional[float] = 0.8
    streaming: Optional[bool] = False
    wav_bytes: Optional[str] = None
    loop: Optional[bool] = False
