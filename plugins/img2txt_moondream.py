import logging
import math
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.console_logging import log_highpower
from utils.gpu_utils import autodetect_device, autodetect_dtype, set_seed
from utils.image_utils import get_image_from_request

def round_up_to_nearest_multiple(n, m):
    return math.ceil(n / m) * m


class VisionRequest(BaseModel):
    image: str
    prompt: Optional[str] = "Describe the image in a few words."
    seed: Optional[int] = -1


class Img2TxtMoondreamPlugin(PluginBase):
    name = "Vision (vikhyatk/moondream2)"
    description = "Image-to-text using Moondream."
    device = autodetect_device()
    dtype = autodetect_dtype(False)
    instance = None

    def __init__(self):        

        model_id = "vikhyatk/moondream2"

        self.dtype = autodetect_dtype(False)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        moondream = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        ).to(
            device=self.device,
            dtype=self.dtype,
        )
        moondream.eval()

        super().__init__()

        self.resources["moondream"] = moondream
        self.resources["tokenizer"] = tokenizer

    async def generate_response(self, image: Image.Image, prompt: str, seed: int = -1):        

        from transformers import AutoModelForCausalLM
        moondream: AutoModelForCausalLM = self.resources["moondream"]
        seed = set_seed(seed)

        moondream.to(
            device=self.device,
            dtype=self.dtype,
        )

        print("Getting response...")
        response = moondream.query(image, prompt, False)
        answer = response.get("answer", "").strip()

        if not answer:
            raise HTTPException(status_code=500, detail="No response")

        return answer
    
    def offload(self):
        moondream = self.resources.get("moondream")
        if moondream:
            moondream.to(
                device="cpu",            
            )


@PluginBase.router.post("/img2txt/moondream", response_class=JSONResponse)
async def vision(req: VisionRequest):
    plugin = None
    try:
        img = get_image_from_request(req.image)

        plugin: Img2TxtMoondreamPlugin = await use_plugin(Img2TxtMoondreamPlugin)

        max_height = 1080
        max_width = 1920

        if img.width > max_width or img.height > max_height:
            aspect_ratio = img.width / img.height
            if img.width > max_width:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
                new_height = round_up_to_nearest_multiple(new_height, 32)
                img = img.resize((new_width, new_height))
            else:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
                new_width = round_up_to_nearest_multiple(new_width, 32)
                img = img.resize((new_width, new_height))


        log_highpower(f"Inspecting image...");
        answer = await plugin.generate_response(img, req.prompt)

        return {"response": answer}
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            release_plugin(Img2TxtMoondreamPlugin)


@PluginBase.router.get("/img2txt/moondream", response_class=JSONResponse)
async def vision_from_url(req: VisionRequest = Depends()):
    return await vision(req)
