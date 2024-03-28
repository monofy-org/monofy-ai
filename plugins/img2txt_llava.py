import logging
from typing import Optional
from fastapi import Depends
from pydantic import BaseModel
from threading import Thread
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.gpu_utils import autodetect_dtype
from utils.image_utils import get_image_from_request


# history item format:
# {"text": "some text", "image": "base64 encoded image"}
class Img2TxtRequest(BaseModel):
    prompt: Optional[str] = "Describe the image."
    history: Optional[list[dict]] = []
    image: Optional[str] = None


class Img2TxtLlavaPlugin(PluginBase):

    name = "img2txt (Llava)"
    description = "Image-to-text generation"
    instance = None
    dtype = autodetect_dtype()

    def __init__(self):
        import torch
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
        )

        super().__init__()

        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

        self.resources["processor"] = LlavaNextProcessor.from_pretrained(model_name)
        self.resources["model"] = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.resources["model"].to(self.device, dtype=self.dtype)

    def bot_streaming(self, req: Img2TxtRequest):
        from transformers import TextIteratorStreamer

        processor = self.resources["processor"]
        model = self.resources["model"]

        if req.image is not None:
            image = get_image_from_request(req.image)
        else:
            # if there's no image uploaded for this turn, look for images in the past turns
            for hist in req.history:
                if hist.get("image") is not None:
                    image = hist.get("image")
                    break

        prompt = f"[INST] <image>\n{req.prompt} [/INST]"
        image = get_image_from_request(req.image)
        inputs = processor(prompt, image, return_tensors="pt").to(
            self.device, self.dtype
        )

        streamer = TextIteratorStreamer(
            processor, skip_special_tokens=True, skip_prompt=True
        )
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text


@PluginBase.router.post("/img2txt/llava", tags=["Image-to-Text"])
async def img2txt(req: Img2TxtRequest):
    plugin: Img2TxtLlavaPlugin = None
    try:
        plugin: Img2TxtLlavaPlugin = await use_plugin(Img2TxtLlavaPlugin)
        text = ""
        for chunk in plugin.bot_streaming(req):
            text += chunk
        return {"response": text}
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin is not None:
            release_plugin(Img2TxtLlavaPlugin)


@PluginBase.router.get("/img2txt/llava", tags=["Image-to-Text"])
async def img2txt_from_url(req: Img2TxtRequest = Depends()):
    return await img2txt(req)
