import logging
import math
from typing import Optional
from fastapi import Depends, HTTPException
from PIL import Image
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.gpu_utils import autodetect_device, autodetect_dtype, set_seed
from utils.image_utils import get_image_from_request


def round_up_to_nearest_multiple(n, m):
    return math.ceil(n / m) * m


class VisionRequest(BaseModel):
    image: str
    prompt: Optional[str] = "Describe the image in a few words."
    seed: Optional[int] = -1


class VisionPlugin(PluginBase):

    name = "vision"
    description = "Vision"
    device = autodetect_device()
    instance = None

    def __init__(self):
        from submodules.moondream.moondream import Moondream
        from transformers import (
            CodeGenTokenizerFast as Tokenizer,
        )

        model_id = "vikhyatk/moondream2"

        self.dtype = autodetect_dtype(False)

        tokenizer = Tokenizer.from_pretrained(model_id)
        moondream = Moondream.from_pretrained(model_id).to(
            device=VisionPlugin.device, dtype=self.dtype
        )
        moondream.eval()

        super().__init__()

        self.resources = {
            "moondream": moondream,
            "tokenizer": tokenizer,
        }

    async def generate_response(self, image: Image.Image, prompt: str, seed: int = -1):
        from submodules.moondream.moondream import Moondream
        from transformers import (
            CodeGenTokenizerFast as Tokenizer,
        )

        moondream: Moondream = self.resources["moondream"]
        tokenizer: Tokenizer = self.resources["tokenizer"]
        seed = set_seed(seed)
        print("Encoding image...")
        image_embeds = moondream.encode_image(image)
        print("Getting response...")
        return moondream.answer_question(image_embeds, prompt, tokenizer)


@PluginBase.router.post("/vision", response_class=JSONResponse)
async def vision(req: VisionRequest):
    print(req.__dict__)
    plugin = None
    try:
        img = get_image_from_request(req.image)

        plugin: VisionPlugin = await use_plugin(VisionPlugin)

        max_size = 768

        if img.width > 1024 or img.height > max_size:
            if img.width > img.height:
                img = img.resize(
                    (
                        max_size,
                        round_up_to_nearest_multiple(
                            img.height * max_size // img.width, 32
                        ),
                    )
                )
            else:
                img = img.resize(
                    (
                        round_up_to_nearest_multiple(
                            img.width * max_size // img.height, 32
                        ),
                        max_size,
                    )
                )

        response = await plugin.generate_response(img, req.prompt)
        return JSONResponse({"response": response})
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            release_plugin(VisionPlugin)


@PluginBase.router.get("/vision", response_class=JSONResponse)
async def vision_from_url(req: VisionRequest = Depends()):
    return await vision(req)
