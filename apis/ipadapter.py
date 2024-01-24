import io
from fastapi import BackgroundTasks, HTTPException, UploadFile
from fastapi.routing import APIRouter
from fastapi.responses import StreamingResponse
from PIL import Image
from utils.image_utils import crop_and_resize, fetch_image
from settings import (
    SD_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_HEIGHT,
    SD_DEFAULT_SCHEDULER,
    SD_DEFAULT_STEPS,
    SD_DEFAULT_WIDTH,
)

router = APIRouter()


@router.post("/ip-adapter")
@router.get("/ip-adapter")
async def ip_adapter(
    background_tasks: BackgroundTasks,
    face_image: UploadFile = None,
    face_image_url: str = None,
    prompt: str = "",
    negative_prompt: str = "",
    steps: int = SD_DEFAULT_STEPS,
    guidance_scale: float = SD_DEFAULT_GUIDANCE_SCALE,
    controlnet_scale: float = 0.8,
    width: int = SD_DEFAULT_WIDTH,
    height: int = SD_DEFAULT_HEIGHT,
    nsfw: bool = False,
    upscale: float = 0,
    upscale_strength: float = 0.65,
    seed: int = -1,
    scheduler: str = SD_DEFAULT_SCHEDULER,
):
    if face_image is not None:
        image_pil = Image.open(io.BytesIO(await face_image.read()))
        image_pil = crop_and_resize(image_pil, width, height)
    elif face_image_url is not None:
        image_pil = fetch_image(face_image_url)
    else:
        return HTTPException(status_code=400, detail="No image or image_url provided")

    from clients import IPAdapterClient

    image_result: Image.Image = await IPAdapterClient.generate(
        face_image=image_pil,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        controlnet_scale=controlnet_scale,
        width=width,
        height=height,
        nsfw=nsfw,
        upscale=upscale,
        upscale_strength=upscale_strength,
        seed=seed,
        scheduler=scheduler,
    )

    image_bytes = io.BytesIO()
    image_result.save(image_bytes, format="png")
    image_bytes.seek(0)

    return StreamingResponse(image_bytes, media_type="image/png")
