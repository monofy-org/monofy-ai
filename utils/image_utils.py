from transformers import AutoImageProcessor, AutoModelForObjectDetection
from diffusers.utils import load_image
import torch
from PIL import ImageDraw
from utils.torch_utils import autodetect_device

device = autodetect_device()


def detect_objects(image_url: str, threshold=0.9):
    image = load_image(image_url)

    # Load the pre-trained image processor and model
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
    model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")

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

    # del image_processor
    # del model
    # torch.cuda.empty_cache()

    return image
