import logging
from PIL import Image
from PIL import ImageDraw
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from modules.plugins import PluginBase, use_plugin

from utils.image_utils import download_image, get_image_from_request


class DetectRequest(BaseModel):
    image: str
    threshold: float = 0.8
    return_image: bool = False


class DetectYOLOSPlugin(PluginBase):

    model_name = "hustvl/yolos-tiny"
    name = f"Object detection ({model_name})"
    description = f"Object detection using {model_name}, returning a list of detected objects and their bounding boxes with optional image."
    instance = None

    def __init__(self):

        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            pipeline,
        )

        super().__init__()

        model_name = self.__class__.model_name

        self.resources["AutoImageProcessor"] = AutoImageProcessor.from_pretrained(
            model_name,
        )

        self.resources["AutoModelForObjectDetection"] = (
            AutoModelForObjectDetection.from_pretrained(
                model_name,
            )
        )

        self.resources["Age Detection"] = pipeline(
            "image-classification", model="dima806/facial_age_image_detection"
        )

    async def detect_objects(
        self, image: Image.Image, threshold: float = 0.8, return_image: bool = False
    ):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        image_processor: AutoImageProcessor = self.resources["AutoImageProcessor"]
        model: AutoModelForObjectDetection = self.resources[
            "AutoModelForObjectDetection"
        ]
        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        detections = []

        # Draw labeled boxes on the image
        draw = ImageDraw.Draw(image) if return_image else None

        for score, name, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            name = model.config.id2label[name.item()]
            box = [round(i) for i in box.tolist()]
            score = round(score.item(), 3)

            item = {"class": name, "score": score, "box": box}

            detections.append(item)

            if not return_image:
                continue

            # Draw the box
            draw.rectangle(box, outline="red", width=2)

            # Display label and confidence
            label_text = f"{name}: {score}"
            draw.text((box[0], box[1]), label_text, fill="red")

        for i, detection in enumerate(detections):
            if detection["class"] == "person":
                age = self.resources["Age Detection"](image.crop(detection["box"]))[0][
                    "label"
                ]
                if "-" in age:
                    age = age.split("-")[1]

                detections[i]["age"] = age.replace("+", "")

        return {
            "detections": detections,
            "image": image if return_image else None,
        }


@PluginBase.router.post("/detect/yolos")
async def detect_yolos(req: DetectRequest):
    try:
        plugin: DetectYOLOSPlugin = await use_plugin(DetectYOLOSPlugin, True)
        img = get_image_from_request(req.image)
        result = await plugin.detect_objects(img, req.threshold, req.return_image)
        return JSONResponse(result)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@PluginBase.router.get("/detect/yolos")
async def detect_from_url(
    req: DetectRequest = Depends(),
):
    try:
        plugin: DetectYOLOSPlugin = await use_plugin(DetectYOLOSPlugin, True)
        img = download_image(req.image)
        result = await plugin.detect_objects(img, req.threshold, req.return_image)
        return JSONResponse(result)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
