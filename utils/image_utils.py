import base64
import io
import logging
import os
from typing import Literal
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from diffusers.utils import load_image
from nudenet import NudeDetector
from PIL import Image, ImageDraw, ImageFilter
from PIL.ExifTags import TAGS, Base

from utils.file_utils import random_filename

nude_detector = NudeDetector()


def set_exif(image: Image.Image, custom_data: str):
    exif_dict = {TAGS[Base.UserComment]: custom_data}
    image.info["exif"] = exif_dict
    return image


def get_image_from_request(
    image: Image.Image | str | os.PathLike | np.ndarray,
    crop: tuple[int, int] = None,
    mirror=False,
    return_path=False,
    format: Literal["cv2", "pillow"] = "pillow",
):
    if isinstance(image, Image.Image):
        return image

    if os.path.exists(image):
        if image.split(".")[-1].lower() not in [
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "gif",
            "webp",
        ]:
            raise ValueError("Invalid image format " + image)

        image = Image.open(image).convert("RGB")

    elif image.startswith("http://") or image.startswith("https://"):
        image = download_image(image)

    else:  # image.startswith("data:image/"):
        # base64 string
        image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")

    # else:
    #     raise ValueError("Invalid image or none provided")

    if crop:
        image: Image.Image = crop_and_resize(image, crop)

    if mirror:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if return_path:
        filename = random_filename("png")
        if format == "cv2":
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, image)
        else:
            image.save(filename, "png")

        return filename
    else:
        if format == "cv2":
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        return image


def crop_and_resize(image: Image.Image, size: tuple[int, int]):
    # get image dimensions
    img_width, img_height = image.size

    # get aspect ratios
    img_aspect_ratio = img_width / img_height
    new_aspect_ratio = size[0] / size[1]

    # if aspect ratios match, return resized image
    if img_aspect_ratio == new_aspect_ratio:
        return image.resize(size)

    # if aspect ratios don't match, crop image
    if img_aspect_ratio > new_aspect_ratio:
        new_width = int(img_height * new_aspect_ratio)
        new_width -= new_width % 32
        offset = (img_width - new_width) // 2
        crop = (offset, 0, img_width - offset, img_height)
    else:
        new_height = int(img_width / new_aspect_ratio)
        new_height -= new_height % 32
        offset = (img_height - new_height) // 2
        crop = (0, offset, img_width, img_height - offset)

    return image.crop(crop).resize(size)


def extend_image(
    image: Image.Image,
    h: int = 128,
    v: int = 128,
    with_mask: bool = False,
    mask_encoding: str = "RGB",
    mask_border=64,
):
    width, height = image.size
    new_image = Image.new("RGB", (int(width + 2 * h), int(height + 2 * v)), "black")

    # Paste the original image in the center of the extended image
    new_image.paste(image, (h, v))

    # Extend the top and bottom sides
    for y in range(v):
        for x in range(width):
            new_image.putpixel((x + v, y), image.getpixel((x, 0)))
            new_image.putpixel((x + v, height + v + y), image.getpixel((x, height - 1)))

    # Extend the left and right sides
    for x in range(h):
        for y in range(height + 2 * v):
            new_image.putpixel((x, y), new_image.getpixel((h, y)))
            new_image.putpixel(
                (width + h + x, y), new_image.getpixel((width + h - 1, y))
            )

    # NOTE: this may need to be decoupled from mask_border
    new_image = new_image.filter(ImageFilter.GaussianBlur(2))
    new_image.paste(image, (h, v))
    # new_image.show()

    if with_mask:
        mask = Image.new(mask_encoding, (new_image.width, new_image.height), "white")
        # draw a white rectangle in the center of the mask
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(
            [
                (h + mask_border, v + mask_border),
                (width + h - mask_border, height + v - mask_border),
            ],
            fill="black",
        )
        mask = mask.resize((width, height))
        new_image = new_image.resize((width, height))
        return new_image, mask

    return new_image.resize((width, height))


def image_to_bytes(img):
    image_bytes = io.BytesIO()
    img.save(image_bytes, format="png")
    image_bytes.seek(0)
    return image_bytes


def sanitize_url(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.netloc.endswith(".gov"):
        raise ValueError("Access to .gov domains is not allowed.")
    return url


def download_image(image_url: str, format: str = "RGB"):
    image_url = sanitize_url(image_url)

    headers = {
        "Referer": image_url.rsplit("/", 1)[0],
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
    }
    response = requests.get(image_url, headers=headers, stream=True, verify=False)
    if response.status_code == 200:
        img = Image.open(response.raw)
        if format is not None:
            img = img.convert(format)
        return img
    else:
        raise Exception(
            f"Failed to download image from {image_url}. Status code: {response.status_code}"
        )


def create_upscale_mask(width, height, aspect_ratio):
    # Create a black image
    img: Image.Image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)

    # Calculate the dimensions of the white box based on the aspect ratio
    box_width = min(width, int(height * aspect_ratio))
    box_height = min(int(width / aspect_ratio), height)

    # Calculate the position of the white box
    x_offset = (width - box_width) // 2
    y_offset = (height - box_height) // 2

    # Draw the white box
    draw.rectangle(
        [x_offset, y_offset, x_offset + box_width, y_offset + box_height],
        outline="white",
        fill="white",
    )

    return img


def fetch_image(image_url: str):
    return load_image(image_url).convert("RGB")


def image_to_base64_no_header(img: Image.Image | np.ndarray):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    image_io = io.BytesIO()
    img.save(image_io, format="PNG")
    return base64.b64encode(image_io.getvalue()).decode("utf-8")


def base64_to_image(base64_string: str):
    return Image.open(io.BytesIO(base64.b64decode(base64_string)))


filtered_nudity = [
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
]


def detect_nudity(nude_detector: NudeDetector, image: Image.Image):
    # create temp file
    image_path = random_filename("png")
    image.save(image_path, format="PNG")
    detections = nude_detector.detect(image_path)
    os.remove(image_path)

    nsfw_detections = [
        detection for detection in detections if detection["class"] in filtered_nudity
    ]
    return len(nsfw_detections) > 0, detections


def censor(
    image: Image.Image,
    nude_detector: NudeDetector = None,
    pre_detected: list = None,
    blackout_instead_of_pixels=False,
):
    if not nude_detector and not pre_detected:
        raise ValueError("Nude detector or pre-detected nudity info must be provided")

    if pre_detected:
        detections = pre_detected
    else:
        _, detections = detect_nudity(nude_detector, image)

    detections = [
        detection for detection in detections if detection["class"] in filtered_nudity
    ]

    # convert PIL image to cv2 image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for detection in detections:
        box = detection["box"]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # change these pixels to pure black

        if blackout_instead_of_pixels:
            img[y : y + h, x : x + w] = (0, 0, 0)
        else:
            # create a 3x3 grid of blurry boxes
            box_w = box_w = int(w // 3)
            box_h = box_h = int(h // 3)
            for i in range(3):
                for j in range(3):
                    # calculate the coordinates of each box
                    box_x = x + i * box_w
                    box_y = y + j * box_h
                    # blur the box region
                    blurred_box = cv2.GaussianBlur(
                        img[box_y : box_y + box_h, box_x : box_x + box_w], (99, 99), 0
                    )
                    # replace the box region with the blurred box
                    img[box_y : box_y + box_h, box_x : box_x + box_w] = blurred_box

    # convert cv2 image to PIL image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    return img, detections


def get_canny_image(image: Image.Image, threshold1: int = 100, threshold2: int = 200):
    logging.info("Performing edge detection")
    from cv2 import Canny

    # convert image to MatLike
    image = np.array(image)

    outline = Canny(image, threshold1, threshold2)

    # convert to PIL image
    outline = Image.fromarray(outline, "L")

    return outline


def resize_keep_aspect_ratio(image: Image.Image, max_size: int = 1024):
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
        new_size = (new_width, new_height)
        return image.resize(new_size)
    elif height > width:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)
        new_size = (new_width, new_height)
        return image.resize(new_size)
    else:
        return image.resize((max_size, max_size))
