import logging
import os
import cv2
from fastapi import BackgroundTasks, HTTPException, Depends
from fastapi.responses import FileResponse
import numpy as np
from numpy import ndarray
import rembg
from typing import Literal, Optional
from PIL import Image
from pydantic import BaseModel
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from modules.plugins import PluginBase, use_plugin, release_plugin
from plugins.txt2model_shap_e import _ply_to_glb
from safetensors.torch import load_file
from utils.file_utils import delete_file, random_filename
from utils.gpu_utils import autodetect_dtype
from utils.image_utils import get_image_from_request
from huggingface_hub import hf_hub_download

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class Img2ModelLGMRequest(BaseModel):
    image: str
    num_inference_steps: Optional[int] = 40
    guidance_scale: Optional[float] = 3.0
    negative_prompt: Optional[str] = ""
    format: Literal["glb", "ply"] = "ply"


class Img2ModelLGMPlugin(PluginBase):

    name = "img2model_lgm"
    description = "Text-to-model generation using LGM"
    dtype = autodetect_dtype(False)
    instance = None

    def __init__(self):

        super().__init__()

        import torch
        from submodules.LGM.core.models import LGM
        from submodules.LGM.mvdream.pipeline_mvdream import MVDreamPipeline
        from submodules.LGM.core.options import Options

        opt = Options(
            input_size=256,
            up_channels=(1024, 1024, 512, 256, 128),  # one more decoder
            up_attention=(True, True, True, False, False),
            splat_size=128,
            output_size=512,  # render & supervise Gaussians at a higher resolution.
            batch_size=8,
            num_views=8,
            gradient_accumulation_steps=1,
            mixed_precision="bf16",
            resume=hf_hub_download(
                repo_id="ashawkey/LGM", filename="model_fp16.safetensors"
            ),
        )

        model = LGM(opt)

        # resume pretrained checkpoint
        if opt.resume is not None:
            if opt.resume.endswith("safetensors"):
                ckpt = load_file(opt.resume, device="cpu")
            else:
                ckpt = torch.load(opt.resume, map_location="cpu")
            model.load_state_dict(ckpt, strict=False)
            logging.info(f"Loaded checkpoint from {opt.resume}")
        else:
            logging.warn("Model randomly initialized, are you sure?")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.half().to(device)
        model.eval()

        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
        proj_matrix[0, 0] = -1 / tan_half_fov
        proj_matrix[1, 1] = -1 / tan_half_fov
        proj_matrix[2, 2] = -(opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = -(opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1

        # load dreams
        pipe_text = MVDreamPipeline.from_pretrained(
            "ashawkey/mvdream-sd2.1-diffusers",  # remote weights
            torch_dtype=torch.float16,
            trust_remote_code=True,
            # local_files_only=True,
        )
        pipe_text = pipe_text.to(device)

        pipe_image = MVDreamPipeline.from_pretrained(
            "ashawkey/imagedream-ipmv-diffusers",  # remote weights
            torch_dtype=torch.float16,
            trust_remote_code=True,
            # local_files_only=True,
        )
        pipe_image = pipe_image.to(device)

        self.resources["LGM"] = model
        self.resources["options"] = opt
        self.resources["proj_matrix"] = proj_matrix
        self.resources["pipe_text"] = pipe_text
        self.resources["pipe_image"] = pipe_image
        self.resources["rembg"] = rembg.new_session()

    def process(
        self,
        filename_noext: str,
        input_image: Image.Image,
        prompt_neg: str = "",
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
    ):
        import torch
        from submodules.LGM.mvdream.pipeline_mvdream import MVDreamPipeline
        from submodules.LGM.core.options import Options
        from submodules.LGM.core.models import LGM

        output_ply_path = f"{filename_noext}.ply"
        bg_remover = self.resources["rembg"]
        model: LGM = self.resources["LGM"]
        opt: Options = self.resources["options"]
        pipe_image: MVDreamPipeline = self.resources["pipe_image"]
        carved_image = rembg.remove(input_image, session=bg_remover)  # [H, W, 4]
        if type(carved_image) is not np.ndarray:
            carved_image = np.array(carved_image)
        mask = carved_image[..., -1] > 0
        image = recenter(carved_image, mask, border_ratio=0.2)
        image = image.astype(np.float32) / 255.0
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        mv_image = pipe_image(
            "",
            image,
            negative_prompt=prompt_neg,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            elevation=0,
        )

        # generate gaussians
        input_image = np.stack(
            [mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0
        )  # [4, 256, 256, 3], float32
        input_image = (
            torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(self.device)
        )  # [4, 3, 256, 256]
        input_image = F.interpolate(
            input_image,
            size=(opt.input_size, opt.input_size),
            mode="bilinear",
            align_corners=False,
        )
        input_image = TF.normalize(
            input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        )

        rays_embeddings = model.prepare_default_rays(self.device, elevation=0)
        input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(
            0
        )  # [1, 4, 9, H, W]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # generate gaussians
                gaussians = model.forward_gaussians(input_image)

            # save gaussians
            model.gs.save_ply(gaussians, output_ply_path)

        return output_ply_path


@PluginBase.router.post(
    "/img2model/lgm", response_class=FileResponse, tags=["3D Model Generation"]
)
async def img2model(
    background_tasks: BackgroundTasks,
    req: Img2ModelLGMRequest,
):
    plugin = None
    filename: str = None

    try:
        image = get_image_from_request(req.image)
        plugin: Img2ModelLGMPlugin = await use_plugin(Img2ModelLGMPlugin)
        filename_noext = random_filename()
        path = plugin.process(
            filename_noext,
            image,
            req.negative_prompt,
            req.num_inference_steps,
            req.guidance_scale,
        )
        if format == "glb":
            path = _ply_to_glb(filename_noext, False)
            return FileResponse(
                path,
                media_type="model/gltf-binary",
                filename=os.path.basename(path),
            )
        else:
            return FileResponse(
                path,
                media_type="model/ply",
                filename=os.path.basename(path),
            )
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin is not None:
            release_plugin(Img2ModelLGMPlugin)
        if filename and os.path.exists(filename):
            background_tasks.add_task(delete_file, filename)


@PluginBase.router.get(
    "/img2model/lgm", response_class=FileResponse, tags=["3D Model Generation"]
)
async def img2model_from_url(
    req: Img2ModelLGMRequest = Depends(),
):
    return await img2model(req)


def recenter(image: ndarray, mask: ndarray, border_ratio: float = 0.2) -> ndarray:
    """recenter an image to leave some empty space at the image border.

    Args:
        image (ndarray): input image, float/uint8 [H, W, 3/4]
        mask (ndarray): alpha mask, bool [H, W]
        border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

    Returns:
        ndarray: output image, float/uint8 [H, W, 3/4]
    """

    return_int = False
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255
        return_int = True

    H, W, C = image.shape
    size = max(H, W)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result = np.ones((size, size, C), dtype=np.float32)
    else:
        result = np.zeros((size, size, C), dtype=np.float32)

    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
        image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA
    )

    if return_int:
        result = (result * 255).astype(np.uint8)

    return result
