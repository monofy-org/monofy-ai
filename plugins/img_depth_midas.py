import logging
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.img_depth_anything import DepthRequest
from utils.image_utils import get_image_from_request, image_to_base64_no_header

DEPTH_MODEL = "DPT_Hybrid"  # DPT_Hybrid, DPT_Large, MiDaS_small supported


class DepthPlugin(PluginBase):

    name = "Depth estimation (MiDaS)"
    description = "Depth estimation using MiDaS model. Returns a depth map image."
    instance = None

    def __init__(self):
        import torch

        super().__init__()
        self.resources["MiDaS"] = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL).to(
            self.device
        )
        self.resources["transforms"] = torch.hub.load("intel-isl/MiDaS", "transforms")

    async def generate_depthmap(self, img: Image.Image):

        import torch
        import cv2
        import numpy as np
        from scipy.signal import medfilt

        cv_image = np.array(img)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        midas = self.resources["MiDaS"]
        transforms = self.resources.get("transforms")

        if DEPTH_MODEL == "DPT_Large" or DEPTH_MODEL == "DPT_Hybrid":
            transform = transforms.dpt_transform
        else:
            transform = transforms.small_transform

        input_batch = transform(img).to(self.device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert prediction to numpy array
        output = prediction.cpu().numpy()

        # Apply a median filter
        filtered_output = medfilt(output, kernel_size=5)

        # Normalize the output to the range 0-255
        normalized_output = filtered_output - np.min(filtered_output)
        normalized_output = normalized_output / np.max(normalized_output) * 255

        # Convert the normalized output to 8-bit format
        formatted = normalized_output.astype(np.uint8)

        # Create an image from the formatted output
        img = Image.fromarray(formatted)

        # release_plugin(DepthPlugin)

        return img


@PluginBase.router.post(
    "/img/depth/midas", response_class=StreamingResponse, tags=["Image Processing"]
)
async def depth_estimation(req: DepthRequest):
    """API route for depth detection"""

    try:

        plugin: DepthPlugin = await use_plugin(DepthPlugin, True)

        image_pil = get_image_from_request(req.image)
        depth_image = await plugin.generate_depthmap(image_pil)

        return {"image": image_to_base64_no_header(depth_image)}
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_plugin(DepthPlugin)


@PluginBase.router.get(
    "/img/depth/midas", response_class=StreamingResponse, tags=["Image Processing"]
)
async def depth_estimation_from_url(req: DepthRequest = Depends()):
    return await depth_estimation(req)
