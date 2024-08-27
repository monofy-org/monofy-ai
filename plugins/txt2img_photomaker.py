import logging
import os
from fastapi import Depends
from typing import Optional
import huggingface_hub
import numpy as np
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.stable_diffusion import StableDiffusionPlugin, format_response
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from submodules.PhotoMaker.photomaker.pipeline_t2i_adapter import (
    PhotoMakerStableDiffusionXLAdapterPipeline,
)
from utils.gpu_utils import autodetect_dtype
from utils.image_utils import get_image_from_request
from utils.stable_diffusion_utils import postprocess


class Txt2ImgPhotoMakerRequest(Txt2ImgRequest):
    adapter_conditioning_scale: Optional[float] = 0.7
    adapter_conditioning_factor: Optional[float] = 0.8
    style_strength_ratio: Optional[float] = 20


class Txt2ImgPhotoMakerPlugin(StableDiffusionPlugin):

    name = "Text-to-image (PhotoMaker)"
    description = "Generate images using PhotoMaker."
    instance = None
    plugins = []

    def __init__(self):
        from submodules.PhotoMaker.photomaker import (
            PhotoMakerStableDiffusionXLAdapterPipeline,
            FaceAnalysis2,
        )
        from diffusers import T2IAdapter

        adapter = T2IAdapter.from_pretrained(
            "TencentARC/t2i-adapter-sketch-sdxl-1.0",
            torch_dtype=autodetect_dtype(),
            variant="fp16",
        )

        face_detector = FaceAnalysis2(
            providers=["CUDAExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        face_detector.prepare(ctx_id=0, det_size=(640, 640))

        super().__init__(PhotoMakerStableDiffusionXLAdapterPipeline, adapter=adapter)

        self.photomaker_ckpt = hf_hub_download(
            repo_id="TencentARC/PhotoMaker-V2",
            filename="photomaker-v2.bin",
            repo_type="model",
        )

        self.resources["adapter"] = adapter.to(self.device)
        self.resources["face_detector"] = face_detector

    def generate(self, req: Txt2ImgPhotoMakerRequest):

        import torch
        from submodules.PhotoMaker.photomaker import FaceAnalysis2, analyze_faces

        face_detector: FaceAnalysis2 = self.resources["face_detector"]

        pipe: PhotoMakerStableDiffusionXLAdapterPipeline = None

        if self.resources.get("pipeline"):
            pipe = self.resources["pipeline"]

        if not self.resources.get("pipeline") or req.model_index != self.model_index:
            self.load_model(req.model_index)
            pipe = self.resources["pipeline"]

            pipe.load_photomaker_adapter(
                os.path.dirname(self.photomaker_ckpt),
                subfolder="",
                weight_name=os.path.basename(self.photomaker_ckpt),
                trigger_word="img",
                pm_version="v2",
            )
            pipe.id_encoder.to(self.device)
            pipe.fuse_lora()
            pipe.unload_lora_weights()
            pipe.enable_model_cpu_offload()

        image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
        input_ids = pipe.tokenizer.encode(req.prompt)

        if image_token_id not in input_ids:
            raise Exception(
                f"Cannot find the trigger word '{pipe.trigger_word}' in text prompt! Please refer to step 2️⃣"
            )

        if input_ids.count(image_token_id) > 1:
            raise Exception(
                f"Cannot use multiple trigger words '{pipe.trigger_word}' in text prompt!"
            )

        start_merge_step = int(req.style_strength_ratio / 100 * req.num_inference_steps)
        if start_merge_step > 30:
            start_merge_step = 30

        input_id_images = [get_image_from_request(req.image, (768, 768))]

        id_embed_list = []

        for img in input_id_images:
            img = np.array(img)
            img = img[:, :, ::-1]
            faces = analyze_faces(face_detector, img)
            if len(faces) > 0:
                id_embed_list.append(torch.from_numpy((faces[0]["embedding"])))

        if len(id_embed_list) == 0:
            raise ValueError("No faces detected")

        id_embeds = torch.stack(id_embed_list)

        req.image = None  # important, this is a doodle sketch image (unused)

        kwargs = dict(
            adapter_conditioning_scale=req.adapter_conditioning_scale,
            adapter_conditioning_factor=req.adapter_conditioning_factor,
            start_merge_step=start_merge_step,
            input_id_images=input_id_images,
            id_embeds=id_embeds,
        )

        return super().generate(
            "txt2img",
            req,
            **kwargs
        )        


@PluginBase.router.post("/txt2img/photomaker", tags=["Image Generation"])
async def txt2img_photomaker(req: Txt2ImgPhotoMakerRequest):
    plugin: Txt2ImgPhotoMakerPlugin = None
    try:
        plugin = await use_plugin(Txt2ImgPhotoMakerPlugin)
        response = await plugin.generate(req)        
    
        return format_response(response)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin is not None:
            release_plugin(Txt2ImgPhotoMakerPlugin)


@PluginBase.router.get("/txt2img/photomaker", tags=["Image Generation"])
async def txt2img_photomaker_from_url(req: Txt2ImgPhotoMakerRequest = Depends()):
    return await txt2img_photomaker(req)
