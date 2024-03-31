import os
import logging
from fastapi import HTTPException
from fastapi.responses import FileResponse
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.file_utils import random_filename
from pydantic import BaseModel


class Txt2ModelShapERequest(BaseModel):
    prompt: str
    guidance_scale: float = 15.0
    num_inference_steps: int = 32
    format: str = "gif"
    frame_size: int = 256


class Txt2ModelShapEPlugin(PluginBase):

    name = "Shap-E"
    description = "Shap-E text-to-3d model generation"
    instance = None

    def __init__(self):

        super().__init__()
        from diffusers import ShapEPipeline

        # openai/shap-e
        pipe: ShapEPipeline = ShapEPipeline.from_pretrained("openai/shap-e").to(
            self.device, dtype=self.dtype
        )
        pipe.enable_model_cpu_offload()

        self.resources["pipeline"] = pipe

    async def generate_shape(
        self,
        req: Txt2ModelShapERequest,
    ):
        from diffusers import ShapEPipeline

        assert format in ["gif", "ply", "glb"], "Format must be 'gif', 'ply', or 'glb'"

        from diffusers.utils import export_to_gif, export_to_ply

        file_path_noext = random_filename()

        kwargs = dict(
            prompt=req.prompt,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            frame_size=req.frame_size,
        )

        pipe: ShapEPipeline = self.resources["pipeline"]

        if format == "ply":
            kwargs["output_type"] = "mesh"
            images = pipe(**kwargs).images[0]

        elif format == "glb" or format == "ply":
            kwargs["output_type"] = "mesh"
            images = pipe(**kwargs).images[0]
            ply_path = f"{file_path_noext}.ply"
            export_to_ply(images, ply_path)
            if format == "glb":
                _ply_to_glb(file_path_noext)
                os.remove(ply_path)

        else:
            images = pipe(**kwargs).images[0]
            export_to_gif(images, f"{file_path_noext}.gif")

        return f"{file_path_noext}.{format}"


@PluginBase.router.post("/shape", tags=["3D Model Generation"])
async def shape(req: Txt2ModelShapERequest):
    plugin = None
    try:
        plugin: Txt2ModelShapEPlugin = await use_plugin(Txt2ModelShapEPlugin)

        file_path = await plugin.generate_shape(req)

        media_type = "image/gif" if req.format == "gif" else "application/octet-stream"
        return FileResponse(file_path, media_type=media_type)

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if plugin:
            release_plugin(Txt2ModelShapEPlugin)


@PluginBase.router.get("/shape", tags=["3D Model Generation"])
async def shape_get(req: Txt2ModelShapERequest):
    return await shape(req)


def _ply_to_glb(file_path_noext: str, rotate=True):
    import trimesh
    import numpy as np

    mesh = trimesh.load(file_path_noext + ".ply")
    if rotate:
        rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh = mesh.apply_transform(rot)
    glb_path = file_path_noext + ".glb"
    mesh.export(glb_path, file_type="glb")
    return glb_path
