from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.file_utils import delete_file, random_filename


class Txt2ModelMesgGPTRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0


class Txt2ModelMeshGPTPlugin(PluginBase):

    name = "Text-to-model (MeshGPT)"
    description = "Generate 3D models from text."
    instance = None
    plugins = []

    def __init__(self):
        super().__init__()

        from meshgpt_pytorch import MeshTransformer

        self.resources["transformer"] = MeshTransformer.from_pretrained(
            "MarcusLoren/MeshGPT-preview"
        ).to(self.device)

    def generate(self, req: Txt2ModelMesgGPTRequest):

        from meshgpt_pytorch import MeshTransformer
        from meshgpt_pytorch import mesh_render

        transformer: MeshTransformer = self.resources["transformer"]

        output = transformer.generate(
            texts=[x.strip() for x in req.prompt.split(",")],
            temperature=req.temperature,
        )

        filename = random_filename("obj")

        mesh_render.save_rendering(filename, [output])

        return filename


@PluginBase.router.post("/txt2model/meshgpt", tags=["3D Model Generation"])
async def txt2model_meshgpt(background_tasks: BackgroundTasks, req: Txt2ModelMesgGPTRequest):
    plugin: Txt2ModelMeshGPTPlugin = None
    filename: str = None
    try:
        plugin = await use_plugin(Txt2ModelMeshGPTPlugin)
        filename = plugin.generate(req)
        return FileResponse(
            filename, filename=filename, media_type="application/octet-stream"
        )

    except Exception:
        return HTTPException(500, "Failed to generate mesh")

    finally:
        release_plugin(Txt2ModelMeshGPTPlugin)
        if filename:
            background_tasks.add_task(delete_file, filename)


@PluginBase.router.get("/txt2model/meshgpt", tags=["3D Model Generation"])
async def txt2model_meshgpt_get(background_tasks: BackgroundTasks, req: Txt2ModelMesgGPTRequest = Depends()):
    return await txt2model_meshgpt(background_tasks, req)
