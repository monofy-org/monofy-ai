import os
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from diffusers.utils import load_image
import torch
from PIL import Image, ImageDraw
from settings import SD_DEFAULT_HEIGHT, SD_DEFAULT_WIDTH

# currently not implemented
DEFAULT_IMAGE_SIZE = (SD_DEFAULT_WIDTH, SD_DEFAULT_HEIGHT)


def is_image_size_valid(image: Image.Image) -> bool:
    return all(dim <= size for dim, size in zip(image.size, DEFAULT_IMAGE_SIZE))


def crop_and_rescale(image: Image, width, height):
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
        offset = (img_width - new_width) // 2
        crop = (offset, 0, img_width - offset, img_height)
    else:
        new_height = int(img_width / new_aspect_ratio)
        offset = (img_height - new_height) // 2
        crop = (0, offset, img_width, img_height - offset)

    cropped_image = image.crop(crop)
    return cropped_image.resize((width, height))


def create_upscale_mask(width, height, aspect_ratio):
    # Create a black image
    img = Image.new("RGB", (width, height), "black")
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


def detect_objects(image_url: str, threshold=0.9):
    image = load_image(image_url)

    # Load the pre-trained image processor and model
    image_processor = AutoImageProcessor.from_pretrained(
        "hustvl/yolos-tiny", cache_dir=os.path.join("models", "YOLOS")
    )
    model = AutoModelForObjectDetection.from_pretrained(
        "hustvl/yolos-tiny", cache_dir=os.path.join("models", "YOLOS")
    )

    # Process the image and get predictions
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    # Draw labeled boxes on the image
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]

        # Draw the box
        draw.rectangle(box, outline="red", width=2)

        # Display label and confidence
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
        draw.text((box[0], box[1]), label_text, fill="red")

    return image
