import logging
import os
from fastapi import Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.file_utils import random_filename
from utils.image_utils import get_image_from_request


class VFusionRequest(BaseModel):
    image: str
    # export_mesh: Optional[bool] = False
    # export_video: Optional[bool] = False


class Img2ModelVFusionPlugin(PluginBase):
    name = "Image-to-model (VFusion3D)"
    description = "Image-to-model using using facebook/vfusion3d"
    instance = None

    def __init__(self):
        super().__init__()

        model = AutoModel.from_pretrained(
            "facebook/vfusion3d", trust_remote_code=True
        )
        self.resources["model"] = model
        self.resources["processor"] = AutoProcessor.from_pretrained(
            "facebook/vfusion3d"
        )

        model.to(self.device)
        model.eval()

    def generate(
        self,
        image: Image.Image,
        # export_mesh: bool = False,
        # export_video: bool = False,
    ):

        model = self.resources["model"]
        processor = self.resources["processor"]

        image, source_camera = processor(image)

        with torch.no_grad():
            output_planes, mesh_path = model(image, source_camera, export_mesh=True)
            return mesh_path
            # if export_mesh:
            #     output_planes, mesh_path = self.model(image, source_camera, export_mesh=True)
            #     return mesh_path
            # elif export_video:
            #     output_planes, video_path = self.model(image, source_camera, export_video=True)
            #     return video_path
            # else:
            #     output_planes = self.model(image, source_camera)
            #     return output_planes.shape


@PluginBase.router.post("/img2model/vfusion")
async def img2model_vfusion(
    req: VFusionRequest,
):
    image = get_image_from_request(req.image)

    if not image:
        raise ValueError("No image provided")

    plugin: Img2ModelVFusionPlugin = None
    file_path: str = None
    try:
        plugin = await use_plugin(Img2ModelVFusionPlugin)
        file_path = plugin.generate(
            image,
            # req.export_mesh,
            # req.export_video,
        )
        return FileResponse(
            file_path,
            media_type="application/octet-stream",
            filename=os.path.basename(file_path),
        )
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        if plugin is not None:
            release_plugin(Img2ModelVFusionPlugin)


@PluginBase.router.get("/img2model/vfusion")
async def img2model_vfusion_get(
    req: VFusionRequest = Depends(),
):
    return await img2model_vfusion(req)
