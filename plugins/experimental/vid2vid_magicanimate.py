from PIL import Image
import cv2
import numpy as np
from modules.plugins import PluginBase
from submodules.MagicAnimate.demo.animate import MagicAnimate
from utils.image_utils import crop_and_resize, get_image_from_request


class Vid2VidMagicAnimatePlugin(PluginBase):
    name = "MagicAnimate"
    description = "Motion transfer with magic animate"
    instance = None

    def __init__(self):
        super().__init__()

        self.resources["MagicAnimate"] = MagicAnimate(
            "submodules/MagicAnimate/configs/prompts/animation.yaml"
        )

    def generate(
        self,
        reference_image,
        motion_sequence_state,
        seed=1,
        steps=25,
        guidance_scale=7.5,
    ):
        animator: MagicAnimate = self.resources["MagicAnimate"]

        # Get image from request
        image: Image.Image = get_image_from_request(reference_image)

        size = min(image.width, image.height)
        image = crop_and_resize(image, (size, size))

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)        

        return animator(
            image,
            motion_sequence_state,
            seed,
            steps,
            guidance_scale,
        )
