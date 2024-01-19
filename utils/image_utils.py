import os
import requests
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from diffusers.utils import load_image
import torch
from PIL import Image, ImageDraw


# currently not implemented
MAX_IMAGE_SIZE = (1024, 1024)


def is_image_size_valid(image: Image.Image) -> bool:
    return all(dim <= size for dim, size in zip(image.size, MAX_IMAGE_SIZE))


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


def fetch_image(url: str):
    return Image.open(requests.get(url, stream=True).raw)


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
