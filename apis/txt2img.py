import logging
from fastapi import BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
import torch
from hyper_tile import split_attention
from settings import (
    SD_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_HEIGHT,
    SD_DEFAULT_SCHEDULER,
    SD_DEFAULT_STEPS,
    SD_DEFAULT_WIDTH,
    SD_USE_HYPERTILE,
    SD_USE_SDXL,
)
from utils.file_utils import delete_file, random_filename
from utils.gpu_utils import load_gpu_task, set_seed, gpu_thread_lock

router = APIRouter()

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
    upscale_strength: float = 0.65,
    controlnet: str = None,
    widen_coef: float = 0,
    seed: int = -1,
    scheduler: str = SD_DEFAULT_SCHEDULER,
    # face_url: str = None,
    # face_landmarks: bool = False,
):
    async with gpu_thread_lock:
        from clients import SDClient

        load_gpu_task("sdxl" if SD_USE_SDXL else "stable diffusion", SDClient)
        # Convert the prompt to lowercase for consistency

        seed = set_seed(seed)

        # if face_url:
        #    face_path = download_to_cache(face_url)
        #    image = cv2.imread(face_path)
        #    faces = face_app.get(image)
        #    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        prompt = prompt.lower()

        if SDClient.schedulers[scheduler]:
            SDClient.pipelines["txt2img"].scheduler = SDClient.schedulers[scheduler]
            SDClient.pipelines["img2img"].scheduler = SDClient.schedulers[scheduler]
            print("Using scheduler " + scheduler)
        else:
            logging.error("Invalid scheduler param: " + scheduler)

        def do_gen():            
            generated_image = SDClient.pipelines["txt2img"](
                prompt=prompt,
                negative_prompt=(
                    "nudity, genitalia, nipples, nsfw"  # none of this unless nsfw=True
                    if not nsfw
                    else "child:1.1, teen:1.1"  # none of this specifically if nsfw=True (weighted to 110%)
                )
                + "watermark, signature, "
                + negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]

            if not upscale and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return generated_image

        def do_upscale(image):
            return SDClient.upscale(
                image=image,
                original_width=width,
                original_height=height,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                controlnet=controlnet,
                upscale_coef=upscale,
                strength=upscale_strength,
                seed=seed,
            )

        def do_widen(image):
            return SDClient.widen(
                image=image,
                width=width * widen_coef,
                height=height,
                aspect_ratio=width / height,
                prompt=prompt,
                negative_prompt=negative_prompt,
                steps=steps,
                seed=seed,
            )

        def process_and_respond(image):
            temp_path = random_filename("png")
            image.save(temp_path, format="PNG")

            if nsfw:
                background_tasks.add_task(delete_file, temp_path)
                return FileResponse(path=temp_path, media_type="image/png")
            else:
                # try:
                # Preprocess the image (replace this with your preprocessing logic)
                # Assuming nude_detector.censor returns the path of the processed image
                processed_image = SDClient.nude_detector.censor(
                    temp_path,
                    [
                        "ANUS_EXPOSED",
                        "MALE_GENITALIA_EXPOSED",
                        "FEMALE_GENITALIA_EXPOSED",
                        "FEMALE_BREAST_EXPOSED",
                    ],
                )
                delete_file(temp_path)
                background_tasks.add_task(delete_file, processed_image)
                return FileResponse(path=processed_image, media_type="image/png")

        if SD_USE_HYPERTILE:
            # split_vae = split_attention(
            #    SDClient.vae,
            #    tile_size=256,
            #    aspect_ratio=1,
            # )
            split_unet = split_attention(
                SDClient.image_pipeline.unet,
                tile_size=256,
                aspect_ratio=1,
            )
            # with split_vae:
            with split_unet:
                generated_image = do_gen()
                if upscale >= 1:
                    generated_image = do_upscale(generated_image)

                return process_and_respond(generated_image)

        else:
            generated_image = do_gen()
            if upscale >= 1:
                generated_image = do_upscale(generated_image)

            return process_and_respond(generated_image)
