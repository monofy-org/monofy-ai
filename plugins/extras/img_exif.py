from fastapi import Depends
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
    data = [str(v) for v in exif.values()]
    print(data)
    return JSONResponse(content={"exif": data})


@router.get("/img/exif", tags=["Image EXIF Tools"])
async def img_exif_get(req: ImgExifRequest = Depends()):
    return await img_exif(req)
