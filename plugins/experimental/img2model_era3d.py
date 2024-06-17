import cv2
import numpy as np
from PIL import Image
from modules.plugins import PluginBase
from rembg import remove

from submodules.Era3D.app import sam_init, sam_segment


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result





class Img2ModelEra3DPlugin(PluginBase):
    name = "img2model_era3d"
    description = "Text-to-model generation using Era3D"
    instance = None

    def __init__(self):
        super().__init__()
        self.resources["SamPredictor"] = sam_init()

    def preprocess(predictor, input_image, chk_group=None, segment=True, rescale=False):
        RES = 1024
        input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
        if chk_group is not None:
            segment = "Background Removal" in chk_group
            rescale = "Rescale" in chk_group
        if segment:
            image_rem = input_image.convert("RGBA")
            image_nobg = remove(image_rem, alpha_matting=True)
            arr = np.asarray(image_nobg)[:, :, -1]
            x_nonzero = np.nonzero(arr.sum(axis=0))
            y_nonzero = np.nonzero(arr.sum(axis=1))
            x_min = int(x_nonzero[0].min())
            y_min = int(y_nonzero[0].min())
            x_max = int(x_nonzero[0].max())
            y_max = int(y_nonzero[0].max())
            input_image = sam_segment(
                predictor, input_image.convert("RGB"), x_min, y_min, x_max, y_max
            )
        # Rescale and recenter
        if rescale:
            image_arr = np.array(input_image)
            in_w, in_h = image_arr.shape[:2]
            out_res = min(RES, max(in_w, in_h))
            ret, mask = cv2.threshold(
                np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY
            )
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            ratio = 0.75
            side_len = int(max_size / ratio)
            padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            padded_image[
                center - h // 2 : center - h // 2 + h,
                center - w // 2 : center - w // 2 + w,
            ] = image_arr[y : y + h, x : x + w]
            rgba = Image.fromarray(padded_image).resize(
                (out_res, out_res), Image.LANCZOS
            )

            rgba_arr = np.array(rgba) / 255.0
            rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
            input_image = Image.fromarray((rgb * 255).astype(np.uint8))
        else:
            input_image = expand2square(input_image, (127, 127, 127, 0))
        return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)
