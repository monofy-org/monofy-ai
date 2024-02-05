import io
from PIL import Image
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRouter
from fastapi import UploadFile, HTTPException
from utils.image_utils import fetch_image

router = APIRouter()


@router.post("/depth")
@router.get("/depth")
async def depth_detection(image_url: str = "", image: UploadFile = None):
    if image_url:
        image_pil = fetch_image(image_url)
    elif image:
        image_pil = Image.open(io.BytesIO(await image.read()))
    else:
        return HTTPException(status_code=400, detail="No image provided")

    from clients.DepthMidasClient import DepthMidasClient

    depth_image = DepthMidasClient.get_instance().generate(image_pil)
    depth_image_bytes = io.BytesIO()
    depth_image.save(depth_image_bytes, format="png")
    depth_image_bytes.seek(0)
    return StreamingResponse(depth_image_bytes, media_type="image/png")
