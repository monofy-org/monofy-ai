import io
from PIL import Image
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from fastapi import UploadFile, HTTPException
from utils.file_utils import download_to_cache

router = APIRouter()


@router.post("/depth")
@router.get("/depth")
async def depth_detection(image_url: str = "", image: UploadFile = None):
    if image_url:
        image_path = download_to_cache(image_url)
        image_pil = Image.open(image_path)
    elif image:
        image_pil = Image.open(io.BytesIO(await image.read()))
    else:
        return HTTPException(status_code=400, detail="No image provided")

    from clients import DepthMidasClient

    depth_image = DepthMidasClient.generate(image_pil)
    depth_image_bytes = io.BytesIO()
    depth_image.save(depth_image_bytes, format="png")
    depth_image_bytes.seek(0)
    return StreamingResponse(depth_image_bytes, media_type="image/png")
