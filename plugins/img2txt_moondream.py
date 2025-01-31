import logging
import math
from typing import Optional

import torch
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
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


class Img2TxtMoondreamPlugin(PluginBase):
    name = "Vision (vikhyatk/moondream2)"
    description = "Image-to-text using Moondream."
    device = autodetect_device()
    dtype = autodetect_dtype(False)
    instance = None

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from submodules.moondream.moondream.torch.moondream import MoondreamModel

        model_id = "vikhyatk/moondream2"

        self.dtype = autodetect_dtype(False)

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

        self.resources = {
            "moondream": moondream,
            "tokenizer": tokenizer,
        }

    async def generate_response(self, image: Image.Image, prompt: str, seed: int = -1):
        from transformers import CodeGenTokenizerFast as Tokenizer

        from submodules.moondream.moondream.torch.moondream import MoondreamModel

        moondream: MoondreamModel = self.resources["moondream"]
        seed = set_seed(seed)

        print("Getting response...")
        response = moondream.query(image, prompt, False)
        answer = response.get("answer", "").strip()

        if not answer:
            raise HTTPException(status_code=500, detail="No response")

        return answer


@PluginBase.router.post("/vision", response_class=JSONResponse)
async def vision(req: VisionRequest):
    print(req.__dict__)
    plugin = None
    try:
        img = get_image_from_request(req.image)

        plugin: Img2TxtMoondreamPlugin = await use_plugin(Img2TxtMoondreamPlugin)

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

        answer = await plugin.generate_response(img, req.prompt)

        return JSONResponse({"response": answer})
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            release_plugin(Img2TxtMoondreamPlugin)


@PluginBase.router.get("/vision", response_class=JSONResponse)
async def vision_from_url(req: VisionRequest = Depends()):
    return await vision(req)
