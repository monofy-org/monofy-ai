import os
import numpy as np
from fastapi import Depends
from fastapi.responses import FileResponse
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from pydantic import BaseModel
from PIL import Image
from plugins.img2model_tsr import Img2ModelTSRPlugin
from plugins.txt2img_depth import Txt2ImgDepthMidasPlugin
from utils.image_utils import image_to_base64_no_header
from pygltflib import GLTF2, Node

INPUT_IMAGE = "res/avatar-depth-tpose.png"
INPUT_SKELETON = "res/skeleton-mixamo-compat.glb"
INPUT_SKELETON_PREFIX = "mixamorig1:"


class Txt2ModelAvatarRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""


class Txt2ModelAvatarPlugin(PluginBase):

    name = "Text-to-Avatar Model"
    description = "Text to 3D avatar model using DepthAnything and TSR."
    instance = None

    def __init__(self):
        super().__init__()

        self.resources["skeleton"] = load_skeleton()

    async def generate(self, req: Txt2ModelAvatarRequest):

        txt2img: Txt2ImgDepthMidasPlugin = await use_plugin(
            Txt2ImgDepthMidasPlugin, True
        )
        depth = Image.open(INPUT_IMAGE)
        img = await txt2img.generate(
            Txt2ImgRequest(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                image=image_to_base64_no_header(depth),
            )
        )

        # DEBUG: save image
        img.save(".cache/avatar.png")

        img2model: Img2ModelTSRPlugin = await use_plugin(Img2ModelTSRPlugin, True)
        filename = await img2model.generate(img)

        gltf = GLTF2().load(filename)
        skeleton = self.resources["skeleton"]
        for mesh in gltf.meshes:
            bind_mesh_weights(mesh, skeleton)
        gltf.save(filename)

        return filename


@PluginBase.router.post("/txt2model/avatar", tags=["Image Generation (text-to-image)"])
async def txt2model_avatar(req: Txt2ModelAvatarRequest):
    plugin = None
    try:
        plugin: Txt2ModelAvatarPlugin = await use_plugin(Txt2ModelAvatarPlugin)
        filename = await plugin.generate(req)
        return FileResponse(
            filename, media_type="model/glb", filename=os.path.basename(filename)
        )
    finally:
        if plugin is not None:
            release_plugin(Txt2ModelAvatarPlugin)


@PluginBase.router.post(
    "/txt2model/avatar/generate", tags=["Image Generation (text-to-image)"]
)
async def txt2model_avatar_generate(req: Txt2ModelAvatarRequest = Depends()):
    return await Txt2ModelAvatarPlugin().generate(req)


def load_skeleton():
    gltf = GLTF2().load(INPUT_SKELETON)
    skeleton = []
    for node in gltf.nodes:
        if node.name.startswith(INPUT_SKELETON_PREFIX):
            skeleton.append(node)
    return skeleton


def bind_mesh_weights(mesh: GLTF2, bones: list[Node]):
    for primitive in mesh.primitives:
        attributes = primitive.attributes
        print("Mesh: ", mesh)
        print("Primitive: ", primitive)
        print("Attributes: ", attributes)

        # Assuming POSITION attribute holds the vertices
        vertices = attributes.POSITION

        weights = []

        for vert in vertices:
            closest_bone = None
            closest_dist = float("inf")
            for i, bone in enumerate(bones):
                dist = np.linalg.norm(vert - bone.translation)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_bone = i           
            
            weights.append(closest_bone)

        primitive.attributes.WEIGHTS_0 = weights

        print("Weights: ", weights)

    return mesh

    




            

            


