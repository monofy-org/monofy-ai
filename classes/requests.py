from pydantic import BaseModel
from typing import Literal, Optional

from settings import (
    SD_DEFAULT_MODEL_INDEX,
    TXT2VID_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_UPSCALE_STRENGTH,
    SD_USE_FREEU,
    SDXL_USE_REFINER,
    TXT2VID_DEFAULT_MODEL_INDEX,
)


class Txt2ImgRequest(BaseModel):
    prompt: Optional[str] = ""
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
    model_index: Optional[int] = TXT2VID_DEFAULT_MODEL_INDEX
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    guidance_scale: float = 2.5
    num_frames: int = 16
    num_inference_steps: int = 6
    fps: float = 12
    seed: int = -1
    interpolate_film: int = 1
    interpolate_rife: int = 1
    fast_interpolate: bool = True
    audio: str = None
