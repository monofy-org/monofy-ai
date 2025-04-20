import base64
import logging
import time
import uuid
from io import BytesIO
from typing import Optional

import torch
import trimesh
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from modules.plugins import PluginBase, release_plugin, use_plugin
from submodules.Hunyuan3D_2.hy3dgen.rembg import BackgroundRemover
from submodules.Hunyuan3D_2.hy3dgen.shapegen.pipelines import (
    Hunyuan3DDiTFlowMatchingPipeline,
)
from submodules.Hunyuan3D_2.hy3dgen.shapegen.postprocessors import (
    DegenerateFaceRemover,
    FaceReducer,
    FloaterRemover,
)
from submodules.Hunyuan3D_2.hy3dgen.texgen.pipelines import Hunyuan3DPaintPipeline
from utils.console_logging import log_loading
from utils.file_utils import random_filename
from utils.gpu_utils import set_seed
from utils.image_utils import get_image_from_request


class Hy3dgenRequest(BaseModel):
    image: Optional[str] = None
    octree_resolution: Optional[int] = 384
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 5.0
    texture: Optional[bool] = True
    face_count: Optional[int] = 40000
    seed: Optional[int] = -1


class ModelWorker:
    def __init__(
        self,
        model_path="tencent/Hunyuan3D-2mini",
        tex_model_path="tencent/Hunyuan3D-2",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        device="cuda",
        enable_tex=False,
    ):
        self.model_path = model_path
        self.device = device
        log_loading("Hy3dgen", model_path)

        self.rembg = BackgroundRemover()
        self.pipeline: Hunyuan3DDiTFlowMatchingPipeline = (
            Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_path,
                subfolder=subfolder,
                use_safetensors=True,
                device=device,
            )
        )
        self.pipeline.enable_flashvdm()
        # self.pipeline_t2i = HunyuanDiTPipeline(
        #     'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
        #     device=device
        # )
        if enable_tex:
            self.pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained(tex_model_path)


class Hy3dgenPlugin(PluginBase):
    name = "hy3dgen"
    description = "3D model generation using Hy3DGen"

    def __init__(self):
        super().__init__()

    def load_model(self):
        if self.resources.get("pipeline"):
            return self.resources["pipeline"]

        pipeline: Hunyuan3DDiTFlowMatchingPipeline = (
            Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2mini",
                subfolder="hunyuan3d-dit-v2-mini-turbo",
                use_safetensors=True,
                device=self.device,
            )
        )
        pipeline.enable_flashvdm()
        # self.pipeline_t2i = HunyuanDiTPipeline(
        #     'Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled',
        #     device=device
        # )

        pipeline_tex = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")

        self.resources["rembg"] = BackgroundRemover()
        self.resources["pipeline"] = pipeline
        self.resources["pipeline_tex"] = pipeline_tex

        return pipeline

    def generate(self, req: Hy3dgenRequest):
        img = get_image_from_request(image=req.image)

        pipeline = self.load_model()
        pipeline_tex = self.resources["pipeline_tex"]
        image = self.resources["rembg"](img)

        if "mesh" in req:
            mesh = trimesh.load(BytesIO(base64.b64decode(req["mesh"])), file_type="glb")
        else:
            seed, generator = set_seed(req.seed, True)

            params = dict(
                image=image,
                generator=generator,
                octree_resolution=req.octree_resolution,
                num_inference_steps=req.num_inference_steps,
                guidance_scale=req.guidance_scale,
                mc_algo="dmc",
                output_type="trimesh",
            )

            mesh = pipeline(**params)[0]

            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=req.face_count)

        if req.texture:
            mesh = pipeline_tex(mesh, image)

        save_path = random_filename("glb")
        mesh.export(save_path)

        torch.cuda.empty_cache()
        return save_path


@PluginBase.router.post("/hy3dgen", tags=["3D Generation"])
async def hy3dgen(req: Hy3dgenRequest):
    plugin: Hy3dgenPlugin = None
    try:
        plugin = await use_plugin(Hy3dgenPlugin)
        result = plugin.generate(req)
        return FileResponse(result)
    except Exception as e:
        logging.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Error generating 3D model")
    finally:
        if plugin:
            release_plugin(Hy3dgenPlugin)
