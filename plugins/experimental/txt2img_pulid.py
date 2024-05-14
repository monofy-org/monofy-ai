from fastapi import Depends
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase, use_plugin, release_plugin
import numpy as np
import torch

from plugins.stable_diffusion import format_response
from submodules.PuLID.pulid import attention_processor as attention
from submodules.PuLID.pulid.pipeline import PuLIDPipeline
from submodules.PuLID.pulid.utils import resize_numpy_image_long, seed_everything
from utils.image_utils import get_image_from_request
from utils.stable_diffusion_utils import postprocess


class Txt2ImgPuLIDPlugin(PluginBase):

    name = "Text-to-Image (PuLID)"
    description = "Text-to-Image using Pure Lightning ID"
    instance = None

    def __init__(self):
        super().__init__()
        self.resources["pipeline"] = PuLIDPipeline()

    @torch.no_grad()
    async def generate_image(self, req: Txt2ImgRequest):
        pipeline: PuLIDPipeline = self.resources["pipeline"]

        id_image = get_image_from_request(req.id_image)
        # supp_images = args[1:4]
        # id_mix = args[4:]
        supp_images = []
        id_mix = []
        (
            prompt,
            neg_prompt,
            scale,
            n_samples,
            seed,
            steps,
            H,
            W,
            id_scale,
            mode,
            id_mix,
        ) = id_mix

        pipeline.debug_img_list = []
        if mode == "fidelity":
            attention.NUM_ZERO = 8
            attention.ORTHO = False
            attention.ORTHO_v2 = True
        elif mode == "extremely style":
            attention.NUM_ZERO = 16
            attention.ORTHO = True
            attention.ORTHO_v2 = False
        else:
            raise ValueError

        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings = pipeline.get_id_embedding(id_image)
            for supp_id_image in supp_images:
                if supp_id_image is not None:
                    supp_id_image = resize_numpy_image_long(supp_id_image, 1024)
                    supp_id_embeddings = pipeline.get_id_embedding(supp_id_image)
                    id_embeddings = torch.cat(
                        (
                            id_embeddings,
                            supp_id_embeddings if id_mix else supp_id_embeddings[:, :5],
                        ),
                        dim=1,
                    )
        else:
            id_embeddings = None

        seed_everything(seed)
        ims = []
        for _ in range(n_samples):
            img = pipeline.inference(
                prompt, (1, H, W), neg_prompt, id_embeddings, id_scale, scale, steps
            )[0]
            ims.append(np.array(img))

        image, json_response = await postprocess(ims[0])
        return format_response(req, json_response)


@PluginBase.router.post("/txt2img/pulid", tags=["Text-to-Image"])
async def txt2img_pulid(req: Txt2ImgRequest):
    plugin: Txt2ImgPuLIDPlugin = None
    try:
        plugin = await use_plugin(Txt2ImgPuLIDPlugin)
        return await plugin.generate_image(req)
    finally:
        if plugin:
            await release_plugin(plugin)


@PluginBase.router.get("/txt2img/pulid", tags=["Text-to-Image"])
async def txt2img_pulid_from_url(req: Txt2ImgRequest = Depends()):
    return await txt2img_pulid(req)
