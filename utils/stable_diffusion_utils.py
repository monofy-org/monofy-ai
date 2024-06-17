from collections import defaultdict
import json
import logging
from math import ceil
import os
from PIL import Image
from PIL import ImageFilter
from attr import has
from fastapi import HTTPException
from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase
from utils.file_utils import ensure_folder_exists
from utils.gpu_utils import autodetect_dtype, set_seed
from utils.image_utils import censor, detect_nudity, image_to_base64_no_header
from huggingface_hub import hf_hub_download
from settings import SD_MIN_INPAINT_STEPS


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


def get_model(repo_or_path: str):

    if os.path.exists(repo_or_path):
        return os.path.abspath(repo_or_path)

    elif repo_or_path.endswith(".safetensors"):

        # see if it is a valid repo/name/file.safetensors
        parts = repo_or_path.split("/")
        if len(parts) == 3:
            repo = parts[0]
            name = parts[1]
            file = parts[2]
            if not file.endswith(".safetensors"):
                raise ValueError(
                    f"Invalid model path {repo_or_path}. Must be a valid local file or hf repo/name/file.safetensors"
                )

            path = os.path.join("models", "Stable-diffusion")

            if os.path.exists(f"{path}/{file}"):
                model_path = f"{path}/{file}"

            else:
                repo_id = f"{repo}/{name}"
                logging.info(f"Fetching {file} from {repo_id}...")
                hf_hub_download(
                    repo_id,
                    filename=file,
                    local_dir=path,
                )
                model_path = os.path.join(path, file)

        else:
            raise FileNotFoundError(f"Model not found at {repo_or_path}")

        return os.path.abspath(model_path)

    return repo_or_path


def load_prompt_lora(pipe, req: Txt2ImgRequest, lora_settings, last_loras=None):
    # if the prompt contains a keyword from favorites, load the LoRA weights
    results = []

    if req.hyper:
        hyper_lora = hf_hub_download(
            "ByteDance/Hyper-SD", "Hyper-SDXL-8steps-CFG-lora.safetensors"
        )
        results.append(hyper_lora)

    for filename, lora_keywords in lora_settings.items():
        for keyword in lora_keywords:
            prompt = req.prompt.lower()
            if keyword.lower() in prompt:
                # pipe._lora_scale = 0.3
                # plugin.pipeline.set_lora_device(plugin.pipeline.device)                
                results.append(filename)
                break

    if last_loras:
        if set(results) == set(last_loras):
            return last_loras

        logging.info("Unloading previous LoRA weights...")
        pipe.unload_lora_weights()

    for filename in results:
        logging.info(f"Loading LoRA: {filename}")
        pipe.load_lora_weights(
            "models/Stable-diffusion/LoRA/",
            weight_name=filename,
            dtype=autodetect_dtype(),
            lora_scale = 0.8,
        )

    return results


async def postprocess(plugin: PluginBase, image: Image.Image, req: Txt2ImgRequest):

    img2img = plugin.resources.get("img2img")
    inpaint = plugin.resources.get("inpaint")
    nude_detector = plugin.resources["NudeDetector"]

    if img2img and req.upscale >= 1:
        if hasattr(plugin, "upscale_with_img2img"):
            image = await plugin.upscale_with_img2img(image, req)
        else:
            logging.warning("Upscaling not supported")
    if inpaint and req.face_prompt:
        faces_image = inpaint_faces(inpaint, image, req)
        if faces_image:
            image = faces_image
        else:
            raise HTTPException(500, "Failed to inpaint faces")

    nsfw, nsfw_detections = detect_nudity(nude_detector, image)
    # yolos_result = await DetectYOLOSPlugin.detect_objects(plugin, image, return_image=False)
    # yolos_detections = yolos_result["detections"]
    yolos_detections = {}

    if not req.nsfw:
        print("censoring")
        image, detections = censor(image, nude_detector, nsfw_detections)

    return image, {
        "images": [image_to_base64_no_header(image)],
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "seed": req.seed,
        "num_inference_steps": req.num_inference_steps,
        "guidance_scale": req.guidance_scale,
        "nsfw": nsfw,
        "objects": yolos_detections,
        "detections": nsfw_detections,
    }


def inpaint_faces(
    pipe, image: Image.Image, req: Txt2ImgRequest, max_steps=5, increment_seed=True
):
    from submodules.adetailer.adetailer.mediapipe import mediapipe_face_mesh

    output = mediapipe_face_mesh(image, confidence=0.3)
    faces_count = len(output.bboxes)

    face_prompts = req.face_prompt.split(",")
    face_prompts = [x.strip() for x in face_prompts]

    if faces_count == 0:
        logging.info("No faces found")
        return image

    logging.info(f"Detected {faces_count} face{ 's' if faces_count != 1 else '' }")

    seed = req.seed
    num_inference_steps = min(12, req.num_inference_steps or 12)

    # sort by size from biggest to smallest
    output.bboxes = sorted(
        output.bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True
    )

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

    # strength = min(1, max(0.1, 5 / num_inference_steps))
    # logging.info("Calculated inpaint strength: " + str(strength))
    strength = 0.4

    min_steps = (
        8 if req.hyper else SD_MIN_INPAINT_STEPS
    )  # hyper needs exactly 8, you can fudge the other bit

    if (req.num_inference_steps or min_steps) * strength < min_steps:
        logging.warning("Increasing steps to prevent artifacts")
        num_inference_steps = ceil(min_steps / strength)

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

        seed, generator = set_seed(seed, True)

        face = face.resize((512, 512))

        kwargs = {
            "prompt": face_prompts[i if i < len(face_prompts) else -1],
            "negative_prompt": req.negative_prompt or "",
            "image": face,
            "mask_image": face_mask,
            "guidance_scale": 2,
            "num_inference_steps": num_inference_steps,
            "strength": strength,
            "generator": generator,
        }

        image2 = pipe(**kwargs).images[0]

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
    if hasattr(pipe, "enable_freeu"):
        pipe.enable_freeu(s1=0.8, s2=0.2, b1=1.05, b2=1.15)


def disable_freeu(pipe):
    if hasattr(pipe, "disable_freeu"):
        pipe.disable_freeu()
