import cv2
import numpy as np
import torch
from PIL import Image

from utils.gpu_utils import autodetect_device

model_type = "DPT_Hybrid" # or DPT_Large, MiDaS_small

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = autodetect_device()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def generate(img):
    cv_image = np.array(img)
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    midas.to(device)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    # Normalize the output to the range 0-255
    normalized_output = output - np.min(output)
    normalized_output = normalized_output / np.max(normalized_output) * 255

    # Ensure values are in the range 0-255
    clipped_output = np.clip(normalized_output, 0, 255)

    formatted = clipped_output.astype("uint8")
    img = Image.fromarray(formatted)
    return img


def offload():
    midas.to("cpu")
    torch.cuda.empty_cache()
