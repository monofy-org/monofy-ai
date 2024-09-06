from fastapi import Depends
from pydantic import BaseModel
import torch
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.image_utils import get_image_from_request


class DetectOwlRequest(BaseModel):
    image: str
    texts: str


class DetectOwlPlugin(PluginBase):
    name = "Object detection (Owl)"
    model_name = "google/owlvit-base-patch32"
    description = description = (
        f"Object detection using {model_name}, returning a list of detected objects and their bounding boxes with optional image."
    )
    instance = None

    def __init__(self):
        super().__init__()

        from transformers import OwlViTProcessor, OwlViTForObjectDetection

        self.resources["processor"] = OwlViTProcessor.from_pretrained(
            "google/owlvit-base-patch32"
        )
        self.resources["model"] = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-base-patch32"
        )

    def generate(self, **kwargs):

        if not kwargs.get("image"):
            raise Exception("You must provide an image.")

        texts: str = kwargs.get("texts", "")
        texts_array = texts.split(",")
        texts_array = [[text.strip() for text in texts_array]]

        if not texts_array:
            raise Exception(
                "Missing texts param. Provide a comma-separated list of objects to detect."
            )

        image = get_image_from_request(kwargs["image"])

        from transformers import OwlViTProcessor, OwlViTForObjectDetection

        processor: OwlViTProcessor = self.resources["processor"]
        model: OwlViTForObjectDetection = self.resources["model"]

        inputs = processor(text=texts_array, images=image, return_tensors="pt")
        outputs = model(**inputs)

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        target_sizes = torch.Tensor([image.size[::-1]])
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = processor.post_process_object_detection(
            outputs=outputs, threshold=0.1, target_sizes=target_sizes
        )

        i = 0  # Retrieve predictions for the first image for the corresponding text queries

        boxes, scores, labels = (
            results[i]["boxes"],
            results[i]["scores"],
            results[i]["labels"],
        )

        results = []

        # Print detected objects and rescaled box coordinates
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {label} with confidence {round(score.item(), 3)} at location {box}"
            )
            results.append(
                {"text": label, "confidence": round(score.item(), 3), "box": box}
            )

        return results


@PluginBase.router.post("/detect/owl", tags=["Object Detection"])
async def detect_owl(req: DetectOwlRequest):
    plugin: DetectOwlPlugin = None
    try:
        plugin = await use_plugin(DetectOwlPlugin)
        return plugin.generate(**req.__dict__)
    except Exception as e:
        print(e)
        return {"error": str(e)}
    finally:
        if plugin is not None:
            release_plugin(DetectOwlPlugin)


@PluginBase.router.get("/detect/owl", tags=["Object Detection"])
async def detect_owl_from_url(req: DetectOwlRequest = Depends()):
    plugin: DetectOwlPlugin = None
    return await plugin.generate(**req.__dict__)
