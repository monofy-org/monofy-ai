import logging
from typing import Optional
from fastapi import Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from modules.plugins import router
from utils.image_utils import get_image_from_request, image_to_base64_no_header, set_exif


class ImgExifRequest(BaseModel):
    image: str
    exif: Optional[str] = None


@router.post("/img/exif", tags=["Image EXIF Tools"])
async def img_exif(req: ImgExifRequest):
    try:
        image = get_image_from_request(req.image)

        response = {}

        if req.exif:
            image = get_image_from_request(req.image)
            image = set_exif(image, req.exif)
            response["image"] = image_to_base64_no_header(image)

        else:
            exif = image.getexif()
            data = [str(v) for v in exif.values()]
            response["exif"] = data
            print(data)

        return JSONResponse(content=response)

    except Exception as e:
        logging.error(e, exc_info=True)
        return JSONResponse(content={"error": "Error processing request"})


@router.get("/img/exif", tags=["Image EXIF Tools"])
async def img_exif_get(req: ImgExifRequest = Depends()):
    return await img_exif(req)
