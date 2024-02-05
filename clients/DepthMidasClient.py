import cv2
import numpy as np
import torch
from PIL import Image
from scipy.signal import medfilt
from clients.ClientBase import ClientBase
from utils.gpu_utils import autodetect_device, load_gpu_task
from settings import DEPTH_MODEL

device = autodetect_device()


class DepthMidasClient(ClientBase):
    def __init__(self):
        super().__init__("depth")        

    def load_model(self):
        midas = torch.hub.load("intel-isl/MiDaS", DEPTH_MODEL)  # depth model
        midas.to(device)

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")  # transforms

        self.models.append(midas)

        if DEPTH_MODEL == "DPT_Large" or DEPTH_MODEL == "DPT_Hybrid":
            self.models.append(transforms.dpt_transform)
        else:
            self.models.append(transforms.small_transform)

    def generate(self, img):

        load_gpu_task("depth", self)

        if len(self.models) == 0:
            self.load_model()

        midas = self.models[0]
        transform = self.models[1]

        cv_image = np.array(img)
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        input_batch = transform(img).to(device)

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
        filtered_output = medfilt(output, kernel_size=3)

        # Normalize the output to the range 0-255
        normalized_output = filtered_output - np.min(filtered_output)
        normalized_output = normalized_output / np.max(normalized_output) * 255

        # Convert the normalized output to 8-bit format
        formatted = normalized_output.astype(np.uint8)

        # Create an image from the formatted output
        img = Image.fromarray(formatted)

        return img
