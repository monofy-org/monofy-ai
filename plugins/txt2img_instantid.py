# !pip install opencv-python transformers accelerate insightface
import logging
from diffusers.models import ControlNetModel
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
import huggingface_hub
import numpy as np
from insightface.app import FaceAnalysis
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS
from classes.pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)
from utils.gpu_utils import autodetect_dtype
from utils.image_utils import get_image_from_request, image_to_bytes


class Txt2ImageInstantIDPlugin(PluginBase):
    def __init__(self):
        import torch

        super().__init__()

        # prepare 'antelopev2' under ./models
        app = FaceAnalysis(
            name="antelopev2",
            root="",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.resources["app"] = app
        app.prepare(ctx_id=0, det_size=(640, 640))

        # load IdentityNet
        self.resources["controlnet"] = ControlNetModel.from_pretrained(
            "InstantX/InstantID", subfolder="ControlNetModel", torch_dtype=torch.float16
        )

        model_name = SD_MODELS[SD_DEFAULT_MODEL_INDEX]

        single_file = model_name.endswith(".safetensors")
        from_model = (
            StableDiffusionXLInstantIDPipeline.from_single_file
            if single_file
            else StableDiffusionXLInstantIDPipeline.from_pretrained
        )
        pipe: StableDiffusionXLInstantIDPipeline = from_model(
            model_name,
            controlnet=self.resources["controlnet"],
            torch_dtype=autodetect_dtype(),
        )
        pipe.cuda()
        ip_adapter_path = huggingface_hub.hf_hub_download(
            "InstantX/InstantID", "ip-adapter.bin"
        )
        pipe.load_ip_adapter_instantid(ip_adapter_path)
        self.resources["pipe"] = pipe

    async def generate(self, req: Txt2ImgRequest):
        import cv2

        image = get_image_from_request(req.image, (req.width, req.height))
        image.save("test.png")

        app: FaceAnalysis = self.resources["app"]
        pipe: StableDiffusionXLInstantIDPipeline = self.resources["pipe"]

        # prepare face emb
        face_info = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        face_info = sorted(
            face_info,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[
            -1
        ]  # only use the maximum face
        face_emb = face_info["embedding"]
        face_kps = draw_kps(image, face_info["kps"])

        image = pipe(
            req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.8,
            ip_adapter_scale=0.8,
            num_inference_steps=req.num_inference_steps,
        ).images[0]

        if req.return_json:
            return {"images": [image]}
        else:
            return StreamingResponse(image_to_bytes(image), media_type="image/png")


@PluginBase.router.post("/txt2img/instantid", tags=["Image Generation (text-to-image)"])
async def txt2img_instantid(req: Txt2ImgRequest):
    plugin = None
    try:
        plugin: Txt2ImageInstantIDPlugin = await use_plugin(Txt2ImageInstantIDPlugin)
        return await plugin.generate(req)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin:
            release_plugin(plugin)


@PluginBase.router.get("/txt2img/instantid", tags=["Image Generation (text-to-image)"])
async def txt2img_instantid_from_url(req: Txt2ImgRequest = Depends()):
    return await txt2img_instantid(req)
