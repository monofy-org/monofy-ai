import logging
from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif, export_to_ply
import numpy as np
import trimesh
from clients.ClientBase import ClientBase
from utils.gpu_utils import (
    load_gpu_task,
)


class ShapeClient(ClientBase):
    def __init__(self):
        super().__init__("shap-e")

    def load_model(self):
        if len(self.models) == 0:
            ClientBase.load_model(
                self,
                ShapEPipeline,
                "openai/shap-e",
            )

    async def generate(
        self,
        prompt: str,
        file_path: str,
        steps: int = 32,
        guidance_scale: float = 15.0,
        format: str = "gif",
    ):
        async with load_gpu_task(self.friendly_name, self):

            self.load_model()

            if format == "gif":
                images = self.models[0](
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    frame_size=256,
                ).images[0]
            elif format == "ply" or format == "glb":
                images = self.models[0](
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=steps,
                    frame_size=256,
                    output_type="mesh",
                ).images[0]
            else:
                logging.error(f"Invalid format: {format}")
                return None

            if format == "ply" or format == "glb":
                ply_path = f"{file_path}.ply"
                export_to_ply(images, ply_path)

                if format == "glb":
                    file_path = f"{file_path}.glb"
                    _export_to_glb(ply_path, file_path)
                else:
                    file_path = ply_path

            else:
                file_path = f"{file_path}.gif"
                export_to_gif(images, file_path)

            return file_path


def _export_to_glb(ply_path, file_path):
    mesh = trimesh.load(ply_path)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(rot)
    mesh_export = mesh.export(file_path, file_type="glb")
    return mesh_export
