from collections import defaultdict
import json
import logging
import os
from PIL import Image
from PIL import ImageFilter
from fastapi import HTTPException
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase
from utils.file_utils import ensure_folder_exists
from utils.gpu_utils import autodetect_dtype, set_seed
from utils.image_utils import censor, detect_nudity


def load_lora_settings():

    ensure_folder_exists("models/Stable-diffusion/LoRA")
    lora_json = "models/Stable-diffusion/LoRA/favorites.json"

    if os.path.exists(lora_json):
        lora_settings = json.load(open(lora_json))
        if not isinstance(lora_settings, dict):
            logging.error("LoRA settings invalid")
            return None
        return lora_settings
    else:
        lora_settings = defaultdict(list)
        lora_settings["your_lora_file.safetensors"] = [
            "your_keyword",
            "another_keyword",
        ]
        # write example to file
        with open(lora_json, "w") as f:
            f.write(json.dumps(lora_settings, indent=4))

        lora_settings = lora_settings

        return lora_settings


def filter_request(req: Txt2ImgRequest):
    words = req.prompt.lower().replace(",", " ").split(" ")
    banned_words = ["baby", "child", "teen", "kid", "underage"]
    nsfw_words = ["nude", "naked", "nudity", "nsfw"]
    banned_nsfw_words = ["boy", "girl", "young", "student"]
    req.negative_prompt = "child, teenager, " + (req.negative_prompt or "")
    for word in words:
        if word in banned_words:
            raise HTTPException(406, "Prohibited prompt")
        for banned in banned_words:
            if banned in word:
                raise HTTPException(406, "Prohibited prompt")
        if not req.nsfw:
            if word in nsfw_words:
                raise HTTPException(406, "NSFW prompt")
            for banned in nsfw_words:
                if banned in word:
                    raise HTTPException(406, "NSFW prompt")
        if word in banned_nsfw_words:
            raise HTTPException(406, "Prohibited NSFW prompt")
        for banned in banned_nsfw_words:
            if banned in word:
                raise HTTPException(406, "Prohibited NSFW prompt")

    return req


def load_prompt_lora(pipe, req, lora_settings):
    # if the prompt contains a keyword from favorites, load the LoRA weights
    for filename, lora_keywords in lora_settings.items():
        for keyword in lora_keywords:
            prompt = req.prompt.lower()
            if keyword.lower() in prompt:
                logging.info(f"Loading LoRA: {filename}")
                pipe.load_lora_weights(
                    "models/Stable-diffusion/LoRA/",
                    weight_name=filename,
                    dtype=autodetect_dtype(),
                )
                # pipe._lora_scale = 0.3
                # plugin.pipeline.set_lora_device(plugin.pipeline.device)

                break


async def postprocess(plugin: PluginBase, image: Image.Image, req: Txt2ImgRequest):

    img2img = plugin.resources.get("img2img")
    inpaint = plugin.resources.get("inpaint")
    nude_detector = plugin.resources["NudeDetector"]

    if img2img and req.upscale >= 1:
        if hasattr(plugin, "upscale_with_img2img"):
            image = plugin.upscale_with_img2img(image, req)
        else:
            logging.warning("Upscaling not supported")
    if inpaint and req.face_prompt is not None:
        faces_image = inpaint_faces(inpaint, image, req)
        if faces_image:
            image = faces_image
        else:
            logging.warning("Inpainting failed")

    nsfw, nsfw_detections = detect_nudity(nude_detector, image)
    # yolos_result = await DetectYOLOSPlugin.detect_objects(plugin, image, return_image=False)
    # yolos_detections = yolos_result["detections"]
    yolos_detections = {}

    if not req.nsfw:
        print("censoring")
        image, detections = censor(image, nude_detector, nsfw_detections)

    return image, {
        "nsfw": nsfw,
        "objects": yolos_detections,
        "detections": nsfw_detections,
    }


def inpaint_faces(
    pipe, image: Image.Image, req: Txt2ImgRequest, max_steps=5, increment_seed=True
):
    from submodules.adetailer.adetailer.mediapipe import mediapipe_face_mesh

    # DEBUG
    # image.save("face-fix-before.png")
    # convert image to black and white    

    # else:
    #    img2img_kwargs["prompt"] = face_prompt

    # black_and_white = image.convert("L").convert("RGB")

    output = mediapipe_face_mesh(image, confidence=0.4)
    faces_count = len(output.bboxes)

    if faces_count == 0:
        logging.info("No faces found")
        return image

    logging.info(f"Detected {faces_count} face{ 's' if faces_count != 1 else '' }")

    seed = req.seed
    num_inference_steps = req.num_inference_steps

    # if req.num_inference_steps * req.strength > max_steps:
    #    num_inference_steps = max_steps // req.strength
    # else:
    #    num_inference_steps = req.num_inference_steps

    # find the biggest face
    # biggest_face = 0
    biggest_face_size = 0
    for i in range(faces_count):
        bbox = output.bboxes[i]
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if size > biggest_face_size:
            biggest_face_size = size
            # biggest_face = i

    # convert bboxes to squares
    for i in range(faces_count):
        bbox = output.bboxes[i]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        diff = abs(width - height)
        if width < height:
            bbox[0] = bbox[0] - diff // 2
            bbox[2] = bbox[2] + diff // 2
        else:
            bbox[1] = bbox[1] - diff // 2
            bbox[3] = bbox[3] + diff // 2
        output.bboxes[i] = bbox

    # Extends boxes in each direction by pixel_buffer.
    # Provides additional context at the cost of quality.
    face_context_buffer = 32

    for i in range(faces_count):
        bbox = output.bboxes[i]
        bbox[0] = bbox[0] - face_context_buffer
        bbox[1] = bbox[1] - face_context_buffer
        bbox[2] = bbox[2] + face_context_buffer
        bbox[3] = bbox[3] + face_context_buffer
        output.bboxes[i] = bbox

    face_mask_blur = 0.05 * max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    for i in range(faces_count):
        # skip if less than 10% of the image size
        if (output.bboxes[i][2] - output.bboxes[i][0]) * (
            output.bboxes[i][3] - output.bboxes[i][1]
        ) < (biggest_face_size * 0.85):
            logging.info(f"Skipping face #{i+1} (background)")
            continue

        mask = output.masks[i]
        face = image.crop(output.bboxes[i])
        face_mask = mask.crop(output.bboxes[i])
        bbox = output.bboxes[i]

        if increment_seed:
            seed = req.seed + i
        else:
            seed = req.seed

        set_seed(seed)

        face.resize((512, 512))

        img2img_kwargs = {
            "prompt": req.face_prompt,
            "image": face,
            "mask_image": face_mask,
            "num_inference_steps": num_inference_steps,
            "strength": 0.6,
        }

        image2 = pipe(**img2img_kwargs).images[0]

        face_mask = face_mask.filter(ImageFilter.GaussianBlur(face_mask_blur))

        # DEBUG
        # if i == biggest_face:
        #    image2.save("face-image2.png")

        image2 = image2.resize((bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # DEBUG
        # if i == biggest_face:
        #    image2.save("face-image2-small.png")

        image.paste(image2, (bbox[0], bbox[1]), mask=face_mask)

    # DEBUG
    # image.save("face-fix-after.png")

    return image


def enable_freeu(pipe):
    pipe.enable_freeu(s1=0.8, s2=0.2, b1=1.05, b2=1.15)


def disable_freeu(pipe):
    pipe.disable_freeu()
