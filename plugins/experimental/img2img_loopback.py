from classes.requests import Txt2ImgRequest
from modules.plugins import router
from utils.image_utils import get_image_from_request

@router.post("/img2img/loopback", tags=["Image Generation"])
async def img2img_loopback(req: Txt2ImgRequest):
    image = get_image_from_request(req.image)

