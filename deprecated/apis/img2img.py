import io
import logging
from fastapi import BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
import torch
from PIL import Image
from hyper_tile import split_attention
from apis.args import ImagePromptKwargs
from modules.plugins import SupportedPipelines, use_plugin, get_resource
from modules.schedulers import schedulers
from settings import (
    SD_DEFAULT_UPSCALE_STRENGTH,
    SD_USE_HYPERTILE,
)
from utils.file_utils import delete_file, random_filename
from utils.gpu_utils import set_seed
from utils.image_utils import crop_and_resize, fetch_image, censor
from utils.stable_diffusion_utils import widen

router = APIRouter()


@router.get("/img2img")
@router.post("/img2img")
async def img2img(
    background_tasks: BackgroundTasks,
    image_kwargs: ImagePromptKwargs ,
    image: UploadFile = None,
    image_url: str = None,
    upscale: float = 0,
    upscale_strength: float = SD_DEFAULT_UPSCALE_STRENGTH,
):
    pipe = await use_plugin(SupportedPipelines.STABLE_DIFFUSION)
    img2img = get_resource(SupportedPipelines.STABLE_DIFFUSION, "img2img")

    if image is not None:
        image_pil = Image.open(io.BytesIO(await image.read()))
    elif image_url is not None:
        image_pil = fetch_image(image_url)
    else:
        raise HTTPException(status_code=400, detail="No image or image_url provided")

    image_pil = crop_and_resize(image_pil, (image_kwargs.width, image_kwargs.height))

    # Convert the prompt to lowercase for consistency

    seed = set_seed(image_kwargs.seed)

    prompt = image_kwargs.prompt.lower()

    if schedulers[image_kwargs.scheduler]:
        img2img.scheduler = schedulers[image_kwargs.scheduler]
        logging.info("Using scheduler " + image_kwargs.scheduler)
    else:
        logging.error("Invalid scheduler param: " + image_kwargs.scheduler)

    async def do_gen():
        with torch.no_grad():
            generated_image = img2img(
                image=image_pil,
                prompt=prompt,
                negative_prompt=(
                    "nudity, genitalia, nipples, nsfw"  # none of this unless nsfw=True
                    if not image_kwargs.nsfw
                    else ""
                )
                + "child:1.1, teen:1.1, watermark, signature, "
                + image_kwargs.negative_prompt,
                num_inference_steps=image_kwargs.num_inference_steps,
                guidance_scale=image_kwargs.guidance_scale,
                width=image_kwargs.width,
                height=image_kwargs.height,
                strength=1,
            ).images[0]

        if not upscale and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_image

    def do_upscale(image):
        return upscale(
            image=image,
            original_width=image_kwargs.width,
            original_height=image_kwargs.height,
            prompt=prompt,
            negative_prompt=image_kwargs.negative_prompt,
            num_inference_steps=image_kwargs.num_inference_steps,
            controlnet=image_kwargs.controlnet,
            upscale_coef=upscale,
            strength=upscale_strength,
            seed=seed,
        )

    def do_widen(image):
        return widen(
            image=image,
            width=image_kwargs.width * 1.25,
            height=image_kwargs.height,
            aspect_ratio=image_kwargs.width / image_kwargs.height,
            prompt=prompt,
            negative_prompt=image_kwargs.negative_prompt,
            num_inference_steps=image_kwargs.num_inference_steps,
            seed=seed,
        )

    def process_and_respond(image):
        temp_path = random_filename("png", True)
        image.save(temp_path, format="PNG")

        if image_kwargs.nsfw:
            background_tasks.add_task(delete_file, temp_path)
            return FileResponse(path=temp_path, media_type="image/png")
        else:
            # try:
            # Preprocess the image (replace this with your preprocessing logic)
            # Assuming nude_detector.censor returns the path of the processed image
            censored_path, detections = censor(temp_path)
            delete_file(temp_path)
            background_tasks.add_task(delete_file, censored_path)
            return FileResponse(path=censored_path, media_type="image/png")

    if SD_USE_HYPERTILE:
        split_vae = split_attention(
            pipe.vae,
            tile_size=256,
            aspect_ratio=1,
        )
        split_unet = split_attention(
            pipe.unet,
            tile_size=256,
            aspect_ratio=1,
        )
        with split_vae:
            with split_unet:
                generated_image = await do_gen()
                if upscale >= 1:
                    generated_image = do_upscale(generated_image)

                return process_and_respond(generated_image)

    else:
        generated_image = await do_gen()
        if upscale >= 1:
            generated_image = do_upscale(generated_image)

        return process_and_respond(generated_image)
