import json
import logging
import os
from collections import defaultdict
from math import ceil

import tqdm.rich
from diffusers import (
    AnimateDiffPipeline,
    FluxPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from fastapi import HTTPException
from huggingface_hub import hf_hub_download
from PIL import Image, ImageFilter

from classes.requests import Txt2ImgRequest
from modules.plugins import PluginBase, use_plugin_unsafe
from plugins.detect_yolos import DetectYOLOSPlugin
from settings import SD_MIN_INPAINT_STEPS
from utils.console_logging import log_loading, log_recycle
from utils.file_utils import ensure_folder_exists
from utils.gpu_utils import autodetect_dtype, set_seed
from utils.image_utils import censor, detect_nudity, image_to_base64_no_header


def load_lora_settings(subfolder: str):
    ensure_folder_exists("models/Stable-diffusion/LoRA/" + subfolder)
    lora_json = os.path.join(
        "models/Stable-diffusion/LoRA/" + subfolder, "favorites.json"
    )

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


def set_lora_strength(
    pipe: (
        StableDiffusionPipeline
        | StableDiffusionXLPipeline
        | AnimateDiffPipeline
        | FluxPipeline
    ),
    lora_strength: float,
):
    active_adapters: list[str] = pipe.get_active_adapters()
    active_adapters = [x for x in active_adapters if x != "lcm-lora"]
    pipe.set_adapters(active_adapters, lora_strength)


def load_prompt_lora(
    pipe: (
        StableDiffusionPipeline
        | StableDiffusionXLPipeline
        | AnimateDiffPipeline
        | FluxPipeline
    ),
    req: Txt2ImgRequest,
    lora_settings,
    last_loras=None,
):
    # if the prompt contains a keyword from favorites, load the LoRA weights
    results = []

    if req.hyper:
        hyper_lora = hf_hub_download(
            "ByteDance/Hyper-SD", "Hyper-SDXL-8steps-CFG-lora.safetensors"
        )
        results.append(hyper_lora)

    for filename, lora_keywords in lora_settings.items():
        keyword: str
        for keyword in lora_keywords:
            prompt = req.prompt.lower()
            if keyword.lower() in prompt:
                results.append(filename)
                break

    if last_loras:
        if set(results) == set(last_loras):
            set_lora_strength(pipe, req.lora_strength)
            log_recycle(f"Reusing LoRA weights: {', '.join(results)}")
            return last_loras

        logging.info("Unloading previous LoRA weights...")
        pipe.unload_lora_weights()

    filename: str
    for filename in results:
        log_loading("LoRA", filename)

        subfolder = (
            ""
            if "XL" in pipe.__class__.__name__
            else "/flux"
            if "Flux" in pipe.__class__.__name__
            else "/sd15"
        )

        pipe.load_lora_weights(
            f"models/Stable-diffusion/LoRA{subfolder}",
            weight_name=filename,
            dtype=autodetect_dtype(),
            # lora_scale=0.8,
            adapter_name=filename.rstrip(".safetensors").replace(".", "_"),
        )
        set_lora_strength(pipe, req.lora_strength)

    # pipe.set_lora_device(results, pipe.device)

    return results


async def postprocess(
    plugin: PluginBase,
    images: Image.Image | list[Image.Image],
    req: Txt2ImgRequest,
    **additional_kwargs,
):
    img2img = plugin.resources.get("img2img") or plugin.resources.get("pipeline")
    inpaint = plugin.resources.get("inpaint") or plugin.resources.get("pipeline")
    nude_detector = plugin.resources["NudeDetector"]

    image_results: list[str] = []

    skip = False

    nsfw_found = False
    all_detections = []

    for image in images:
        yolos: DetectYOLOSPlugin = use_plugin_unsafe(DetectYOLOSPlugin)
        yolos_result = await yolos.detect_objects(image)
        yolos_detections: dict = yolos_result["detections"]
        
        for d in yolos_detections:
            age = d.get("age")
            if not age or str(age).startswith("0"):
                continue

            score = d.get("score")
            if score < 0.9:
                continue

            print(f"Detected person (guessing age {age}), score = {score}")

            # The age detector is already skewed toward guessing lower so 18+ here is good.
            # It detects most poeple in their 20's as 16.
            # 01 and 02 are frequently attributed to non-people. This could use more work.
            if age and int(age) < 18:
                logging.error("Person under 18 detected")
                skip = True
                break

        if skip:
            skip = False
            continue

        if img2img and req.upscale >= 1 and req.num_inference_steps > 0:
            if hasattr(plugin, "upscale_with_img2img"):
                image = await plugin.upscale_with_img2img(image, req)
            else:
                logging.warning("Upscaling not supported")
        if inpaint and req.face_prompt:
            faces_image = inpaint_faces(inpaint, image, req, **additional_kwargs)
            if faces_image:
                image = faces_image
            else:
                raise HTTPException(500, "Failed to inpaint faces")
        
        nsfw, nsfw_detections = detect_nudity(nude_detector, image)

        all_detections.extend(nsfw_detections)

        if nsfw:
            nsfw_found = True

        if not req.nsfw:
            image, detections = censor(image, nude_detector, nsfw_detections)

        image_results.append(image_to_base64_no_header(image))

    return image, {
        "images": image_results,
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "seed": req.seed,
        "num_inference_steps": req.num_inference_steps,
        "guidance_scale": req.guidance_scale,
        "nsfw": nsfw_found,
        "objects": yolos_detections,
        "detections": all_detections,
    }


def inpaint_faces(
    pipe,
    image: Image.Image,
    req: Txt2ImgRequest,
    max_steps=5,
    increment_seed=True,
    **additional_kwargs,
):
    from submodules.adetailer.adetailer.mediapipe import mediapipe_face_mesh

    output = mediapipe_face_mesh(image, confidence=0.3)
    faces_count = len(output.bboxes)

    face_prompts = req.face_prompt.split(",")
    face_prompts = [x.strip() for x in face_prompts]

    if faces_count == 0:
        logging.info("No faces found")
        return image

    logging.info(f"Detected {faces_count} face{'s' if faces_count != 1 else ''}")

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
            logging.info(f"Skipping face #{i + 1} (background)")
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

        pipe.progress_bar = tqdm.rich.tqdm

        image2 = pipe(**kwargs, **additional_kwargs).images[0]

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


def manual_offload(pipe):
    if not pipe:
        return

    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()
    for attr_name in dir(pipe):
        if attr_name == "__class__":
            continue
        try:
            attr = getattr(pipe, attr_name)
        except Exception:
            continue
        if attr and hasattr(attr, "to") and callable(attr.to):
            try:
                attr.to(device="cpu")
                logging.debug(f"Offloaded {attr_name} to CPU")
            except Exception as e:
                logging.warning(f"Failed to offload {attr_name}: {e}")
