import cv2
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from diffusers.utils import load_image
import torch
from PIL import ImageDraw
from PIL.Image import Image
from settings import SD_DEFAULT_HEIGHT, SD_DEFAULT_WIDTH
from utils.file_utils import import_model
from utils.gpu_utils import autodetect_device, autodetect_dtype
from nudenet import NudeDetector

# currently not implemented
DEFAULT_IMAGE_SIZE = (SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT)

nude_detector = NudeDetector()


def is_image_size_valid(image: Image) -> bool:
    return all(dim <= size for dim, size in zip(image.size, DEFAULT_IMAGE_SIZE))


def crop_and_resize(image: Image, width, height):
    # get image dimensions
    img_width, img_height = image.size

    # get aspect ratios
    img_aspect_ratio = img_width / img_height
    new_aspect_ratio = width / height

    # if aspect ratios match, return resized image
    if img_aspect_ratio == new_aspect_ratio:
        return image.resize((width, height))

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

    cropped_image: Image = image.crop(crop)
    cropped_image = cropped_image.resize((width, height))
    return cropped_image


def create_upscale_mask(width, height, aspect_ratio):
    # Create a black image
    img: Image = Image.new("RGB", (width, height), "black")
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
    return load_image(image_url)


YOLOS_MODEL = "hustvl/yolos-tiny"
image_processor: AutoImageProcessor = None
model: AutoModelForObjectDetection = None


def detect_objects(image_url: str, draw_image=False, threshold=0.8):
    global image_processor
    global model

    image = load_image(image_url)

    if image_processor is None:
        image_processor = import_model(
            AutoImageProcessor,
            YOLOS_MODEL,
            set_variant_fp16=False,
            allow_fp16=True,
            allow_bf16=False,            
        )

    if model is None:
        model = import_model(
            AutoModelForObjectDetection,
            YOLOS_MODEL,
            set_variant_fp16=False,
            allow_fp16=True,
            allow_bf16=False,            
        )

    # Process the image and get predictions
    inputs = image_processor(images=image, return_tensors="pt").to(
        autodetect_device(), dtype=autodetect_dtype(False)
    )
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    objects = []

    # Draw labeled boxes on the image
    draw = ImageDraw.Draw(image) if draw_image else None

    for score, name, box in zip(results["scores"], results["labels"], results["boxes"]):
        name = model.config.id2label[name.item()]
        box = [round(i) for i in box.tolist()]
        score = round(score.item(), 3)

        item = {"class": name, "score": score, "box": box}
        objects.append(item)

        if not draw_image:
            continue

        # Draw the box
        draw.rectangle(box, outline="red", width=2)

        # Display label and confidence
        label_text = f"{name}: {score}"
        draw.text((box[0], box[1]), label_text, fill="red")

    return image, objects


filtered_nudity = [
    "ANUS_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
]


def detect_nudity(image_url: str):
    detections = nude_detector.detect(image_url)
    nsfw_detections = [
        detection for detection in detections if detection["class"] in filtered_nudity
    ]
    return len(nsfw_detections) > 0, detections


def censor(temp_path: str, pre_detected: list = None, blackout_instead_of_pixels=False):
    detections = pre_detected if pre_detected is not None else detect_nudity(temp_path)

    detections = [
        detection for detection in detections if detection["class"] in filtered_nudity
    ]

    img = cv2.imread(temp_path)

    for detection in detections:
        box = detection["box"]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # change these pixels to pure black

        if blackout_instead_of_pixels:
            img[y : y + h, x : x + w] = (0, 0, 0)
        else:
            # create a 3x3 grid of blurry boxes
            for i in range(3):
                for j in range(3):
                    # calculate the coordinates of each box
                    box_x = x + int(i * w // 3)
                    box_y = y + int(j * h // 3)
                    box_w = int(w // 3)
                    box_h = int(h // 3)
                    # blur the box region
                    blurred_box = cv2.GaussianBlur(
                        img[box_y : box_y + box_h, box_x : box_x + box_w], (99, 99), 0
                    )
                    # replace the box region with the blurred box
                    img[box_y : box_y + box_h, box_x : box_x + box_w] = blurred_box

    out_path = temp_path.replace(".png", "_censored.png")
    cv2.imwrite(out_path, img)

    return out_path, detections
