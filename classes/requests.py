from pydantic import BaseModel
from typing import Optional

from settings import (
    SD_DEFAULT_GUIDANCE_SCALE,    
    SD_DEFAULT_UPSCALE_STRENGTH,
    SD_USE_FREEU,
    SD_USE_SDXL,
)


class Txt2ImgRequest(BaseModel):
    prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 768 if SD_USE_SDXL else 512
    height: Optional[int] = 768 if SD_USE_SDXL else 512
    guidance_scale: Optional[float] = SD_DEFAULT_GUIDANCE_SCALE
    num_inference_steps: Optional[int] = None
    seed: Optional[int] = -1
    model_index: Optional[int] = 0
    scheduler: Optional[str] = None
    nsfw: Optional[bool] = False
    face_prompt: Optional[str] = None
    upscale: Optional[float] = 0
    strength: Optional[float] = SD_DEFAULT_UPSCALE_STRENGTH
    auto_lora: Optional[bool] = True
    freeu: Optional[bool] = SD_USE_FREEU
    return_json: Optional[bool] = False
    image: Optional[str] = None


class Txt2VidRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    guidance_scale: float = 2.0
    num_frames: int = 16
    num_inference_steps: int = 6
    fps: float = 12
    seed: int = -1
    interpolate: int = 1
    fast_interpolate: int = 1
    audio: str = None
