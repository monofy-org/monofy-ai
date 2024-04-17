from fastapi.responses import JSONResponse
from pydantic import BaseModel
from modules.plugins import router
from utils.image_utils import get_image_from_request


class ImgExifRequest(BaseModel):
    image: str


@router.post("/img/exif", tags=["Image EXIF Tools"])
async def img_exif(req: ImgExifRequest):
    image = get_image_from_request(req.image)
    exif = image.getexif()
    return JSONResponse(content={"exif": exif})
