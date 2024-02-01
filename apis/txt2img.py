import asyncio
import base64
import gc
import logging
import time
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRouter
import torch
from hyper_tile import split_attention
from settings import (
    SD_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_HEIGHT,
    SD_DEFAULT_SCHEDULER,
    SD_DEFAULT_STEPS,
    SD_DEFAULT_WIDTH,
    SD_MODELS,
    SD_USE_HYPERTILE,
    SD_USE_SDXL,
    SD_FIX_FACES,
)
from utils.file_utils import delete_file, random_filename
from utils.gpu_utils import load_gpu_task, set_seed, gpu_thread_lock
from utils.misc_utils import print_completion_time
from utils.image_utils import detect_nudity, detect_objects, censor

router = APIRouter()


def progress(_, step_index, timestep, callback_kwargs):
    if step_index > 3:
        latent = callback_kwargs["latents"][0][0][0][0]
        if step_index:
            print(step_index, latent)
        # SDClient.pipelines["txt2img"]

    return callback_kwargs


@router.get("/txt2img/models")
async def txt2img_models():
    models = [
        model.replace("\\", "/").split("/")[-1].split(".")[0] for model in SD_MODELS
    ]
    return JSONResponse(content=models)


@router.post("/txt2img")
@router.get("/txt2img")
async def txt2img(
    background_tasks: BackgroundTasks,
    prompt: str,
    negative_prompt: str = "",
    steps: int = SD_DEFAULT_STEPS,
    guidance_scale: float = SD_DEFAULT_GUIDANCE_SCALE,
    width: int = SD_DEFAULT_WIDTH,
    height: int = SD_DEFAULT_HEIGHT,
    nsfw: bool = False,
    upscale: float = 0,
    strength: float = 0.65,
    controlnet: str = None,
    # widen_coef: float = 0,
    model_index: int = 0,
    seed: int = -1,
    scheduler: str = SD_DEFAULT_SCHEDULER,
    # face_url: str = None,
    # face_landmarks: bool = False,
    return_json: bool = False,
    fix_faces=SD_FIX_FACES,
):
    logging.info(f"[txt2img] {prompt}")
    async with gpu_thread_lock:
        from clients import SDClient

        await asyncio.sleep(0.1)

        load_gpu_task("sdxl" if SD_USE_SDXL else "stable diffusion", SDClient)
        # Convert the prompt to lowercase for consistency

        SDClient.load_model(SD_MODELS[model_index])

        logging.info(f"Using model {SD_MODELS[model_index]}")

        seed = set_seed(seed)

        # if face_url:
        #    face_path = download_to_cache(face_url)
        #    image = cv2.imread(face_path)
        #    faces = face_app.get(image)
        #    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        if SDClient.schedulers[scheduler]:
            SDClient.pipelines["txt2img"].scheduler = SDClient.schedulers[scheduler]
            logging.info("Using scheduler " + scheduler)
        else:
            logging.error("Invalid scheduler param: " + scheduler)

        start_time = time.time()
        prompt = prompt.lower()

        def do_gen():
            with torch.no_grad():
                generated_image = SDClient.pipelines["txt2img"](
                    prompt=prompt,
                    negative_prompt=(
                        "nudity, genitalia, nipples, nsfw"  # none of this unless nsfw=True
                        if not nsfw
                        else ""
                    )
                    + "child:1.1, teen:1.1, watermark, signature, "
                    + negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    # callback_on_step_end=progress,
                ).images[0]

            return generated_image

        def do_upscale(image):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()            

            SDClient.pipelines["img2img"].scheduler = SDClient.schedulers[scheduler]
            return SDClient.upscale(
                image=image,
                original_width=width,
                original_height=height,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                controlnet=controlnet,
                upscale_coef=upscale,
                strength=strength,
                seed=seed,
            )

        def do_widen(image):
            SDClient.pipelines["img2img"].scheduler = SDClient.schedulers[scheduler]
            return SDClient.widen(
                image=image,
                width=width * 1.25,
                height=height,
                aspect_ratio=width / height,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed,
            )

        def process_image(image):
            if fix_faces:
                image = SDClient.fix_faces(
                    image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    width=512,
                    height=512,
                    strength=0.5,
                )

            temp_path = random_filename("png")
            image.save(temp_path, format="PNG")

            print_completion_time(start_time, "txt2img")

            detect_start_time = time.time()
            _, objects = detect_objects(temp_path, False)
            is_nsfw, detections = detect_nudity(temp_path)

            print_completion_time(detect_start_time, "Detections")

            if is_nsfw:
                logging.warn("NSFW image detected!")

            print(objects)
            print(detections)

            properties = {
                "nsfw": is_nsfw,
                "objects": objects,
                "detections": detections,
            }

            if nsfw:
                # skip processing
                return temp_path, properties
            else:
                censored_path, detections = censor(temp_path, detections)
                background_tasks.add_task(delete_file, temp_path)

                return censored_path, properties

        if SD_USE_HYPERTILE:
            split_vae = split_attention(
                SDClient.image_pipeline.vae,
                tile_size=256,
                aspect_ratio=1,
            )
            split_unet = split_attention(
                SDClient.image_pipeline.unet,
                tile_size=256,
                aspect_ratio=1,
            )
            with split_vae:
                with split_unet:
                    generated_image = do_gen()
                    if upscale >= 1:
                        generated_image = do_upscale(generated_image)

                    image_path, properties = process_image(generated_image)

        else:
            generated_image = do_gen()
            if upscale >= 1:
                generated_image = do_upscale(generated_image)

            image_path, properties = process_image(generated_image)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()        

        background_tasks.add_task(delete_file, image_path)

        base64Image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

        if return_json:
            response = {
                "images": [base64Image],
                "model": SD_MODELS[model_index]
                .replace("\\", "/")
                .split("/")[-1]
                .split(".")[0],
                "seed": seed,
                "fix_faces": fix_faces,
                "nsfw": properties["nsfw"],
                "objects": properties["objects"],
                "detections": properties["detections"],
            }
            return JSONResponse(content=response)
        else:
            return FileResponse(path=image_path, media_type="image/png")
