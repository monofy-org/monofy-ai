from pydantic import BaseModel
from typing import Literal, Optional

from settings import (
    SD_DEFAULT_LORA_STRENGTH,
    SD_DEFAULT_MODEL_INDEX,
    TXT2VID_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_UPSCALE_STRENGTH,
    SD_USE_FREEU,
    SDXL_USE_REFINER,
    TXT2VID_DEFAULT_MODEL_INDEX,
)


class Txt2ImgRequest(BaseModel):
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
    prompt: str = ""
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 576
    height: Optional[int] = 576
    guidance_scale: Optional[float] = TXT2VID_DEFAULT_GUIDANCE_SCALE
    lora_strength: Optional[float] = 0.8
    num_frames: Optional[int] = 16
    num_inference_steps: Optional[int] = 8
    nsfw: Optional[bool] = False
    fps: Optional[float] = 6
    seed: Optional[int] = -1
    interpolate_film: Optional[int] = 0
    interpolate_rife: Optional[int] = 2
    fast_interpolate: Optional[bool] = False
    audio: Optional[str] = None
    model_index: Optional[int] = TXT2VID_DEFAULT_MODEL_INDEX
    clip_index: Optional[int] = None
    motion_adapter: Optional[Literal["animatediff", "animatelcm"]] = "animatelcm"
    scheduler: Optional[Literal["euler_a", "lcm"]] = None
    use_lightning: Optional[bool] = False
    


class ModelInfoRequest(BaseModel):
    model_index: int = 0
