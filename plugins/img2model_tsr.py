import logging
import os
import rembg
from pydantic import BaseModel
from typing import Literal
from fastapi import BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import trimesh
from modules.plugins import PluginBase, release_plugin, use_plugin
from submodules.TripoSR.tsr.utils import resize_foreground
from utils.file_utils import delete_file, random_filename
from utils.image_utils import get_image_from_request


class Img2ModelTSRRequest(BaseModel):
    image: str
    format: Literal["glb", "obj"] = "glb"
    foreground_ratio: float = (0.85,)


class Img2ModelTSRPlugin(PluginBase):

    name = "Txt2ModelTSR"
    description = "Text to Model TSR"
    instance = None

    def __init__(self):
        super().__init__()

        from submodules.TripoSR.tsr.system import TSR

        model: TSR = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )

        model.renderer.set_chunk_size(8192)

        model.to(self.device)

        self.resources["model"] = model
        self.resources["rembg"] = rembg.new_session()

    async def generate(
        self,
        image: Image.Image,
        format: Literal["glb", "obj"] = "glb",
        foreground_ratio: float = 0.85,
    ):
        import torch
        from submodules.TripoSR.tsr.system import TSR

        model: TSR = self.resources["model"]

        # Remove background
        img = np.array(image)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = rembg.remove(img, alpha_matting=True, session=self.resources["rembg"])

        img = resize_foreground(img, foreground_ratio)
        img = np.array(img).astype(np.float32) / 255.0
        img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4]) * 0.5

        img = Image.fromarray((img * 255.0).astype(np.uint8))

        with torch.no_grad():
            scene_codes = model([img], device=self.device)

        meshes: list = model.extract_mesh(scene_codes, resolution=256)

        filename = random_filename(format)

        mesh: trimesh.Trimesh = meshes[0]

        # rotate trimesh by 90deg on both the x and z axis
        matrix = trimesh.transformations.rotation_matrix(
            np.radians(-90), [1, 0, 0]
        ).dot(trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1]))
        mesh.apply_transform(matrix)

        print("Mesh: ", mesh)

        mesh.export(filename, format)

        return filename


@PluginBase.router.post(
    "/img2model/tsr", response_class=FileResponse, tags=["3D Model Generation"]
)
async def img2model(
    background_tasks: BackgroundTasks,
    req: Img2ModelTSRRequest,
):
    plugin = None
    filename = None

    try:
        plugin: Img2ModelTSRPlugin = await use_plugin(Img2ModelTSRPlugin)

        image = get_image_from_request(req.image)

        filename = await plugin.generate(image, req.format)

        return FileResponse(
            filename,
            media_type=f"model/{req.format}",
            filename=os.path.basename(filename),
        )

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if plugin is not None:
            release_plugin(Img2ModelTSRPlugin)
        if filename and os.path.exists(filename):
            background_tasks.add_task(delete_file, filename)


@PluginBase.router.get(
    "/img2model/tsr", response_class=FileResponse, tags=["3D Model Generation"]
)
async def img2model_from_url(
    background_tasks: BackgroundTasks,
    req: Img2ModelTSRRequest = Depends(),
):
    return await img2model(background_tasks, req)
