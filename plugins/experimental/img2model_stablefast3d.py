import logging
from typing import Literal, Optional
from fastapi import Depends
from pydantic import BaseModel
import rembg
import torch
from PIL import Image
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.file_utils import random_filename
from utils.image_utils import get_image_from_request


class StableFast3DRequest(BaseModel):
    image: str
    foreground_ratio: Optional[int] = 0.85
    texture_resolution: Optional[int] = 1024
    remesh_option: Optional[Literal["none", "triangle", "quad"]] = "none"


class Img2ModelStableFast3DPlugin(PluginBase):
    name = "Img2Model StableFast3D"
    description = "Img2Model StableFast3D"
    instance = None

    def __init__(self):

        super().__init__()

        from submodules.sf3d.sf3d.system import SF3D

        model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        model.to(self.device)
        model.eval()

        self.resources["rembg"] = rembg.new_session()
        self.resources["model"] = model

    def generate(
        self,
        image: Image.Image,
        foreground_ratio=0.85,
        texture_resolution=1024,
        remesh_option: Literal["none", "triangle", "quad"] = "none",
    ):
        from submodules.sf3d.sf3d.utils import remove_background, resize_foreground

        rembg_session = self.resources["rembg"]
        model = self.resources["model"]

        image = remove_background(image.convert("RGBA"), rembg_session)

        image = resize_foreground(image, foreground_ratio)

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mesh, glob_dict = model.run_image(
                    image,
                    bake_resolution=texture_resolution,
                    remesh=remesh_option,
                )
        print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

        out_mesh_path = random_filename("glb")
        mesh.export(out_mesh_path, include_normals=True)

        return out_mesh_path


@PluginBase.router.post("/img2model/stablefast3d")
async def img2model_stablefast3d(
    req: StableFast3DRequest,
):
    image = get_image_from_request(req.image)

    if not image:
        raise ValueError("No image provided")

    plugin: Img2ModelStableFast3DPlugin = None
    try:
        plugin = await use_plugin(Img2ModelStableFast3DPlugin)
        response = plugin.generate(
            image,
            req.foreground_ratio,
            req.texture_resolution,
            req.remesh_option,
        )
        return response
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin is not None:
            release_plugin(Img2ModelStableFast3DPlugin)


@PluginBase.router.get("/img2model/stablefast3d")
async def img2model_stablefast3d_get(
    req: StableFast3DRequest = Depends(),
):
    return await img2model_stablefast3d(req)
