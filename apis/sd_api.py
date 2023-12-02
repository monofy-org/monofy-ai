import os
from datetime import datetime
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    # EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)
from utils.torch_utils import autodetect_device
from PIL import Image
from nudenet import NudeDetector
from settings import (
    SD_MODEL,
    SD_USE_SDXL,
    SD_USE_MODEL_VAE,
    SD_DEFAULT_STEPS,
    SD_DEFAULT_GUIDANCE_SCALE,
)

nude_detector = NudeDetector()


def sd_api(app: FastAPI):
    device = autodetect_device()
    print(f"Stable Diffusion using device: {device}")
    # Load the pretrained model

    pipeline_type = (
        StableDiffusionXLPipeline if SD_USE_SDXL else StableDiffusionPipeline
    )

    pipeline = pipeline_type.from_single_file(
        SD_MODEL, variant="fp16", load_safety_checker=False, torch_dtype=torch.float16
    )
    pipeline.to(device)

    if SD_USE_MODEL_VAE:
        vae = AutoencoderKL.from_single_file(
            SD_MODEL, variant="fp16", torch_dtype=torch.float16
        ).to(device)

        pipeline.vae = vae

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    text_to_image_pipe = AutoPipelineForText2Image.from_pipe(pipeline)
    # image_to_image_pipe = AutoPipelineForImage2Image.from_pipe(pipeline)

    CACHE_DIR = ".cache"
    MAX_IMAGE_SIZE = (1024, 1024)

    def is_image_size_valid(image: Image.Image) -> bool:
        return all(dim <= size for dim, size in zip(image.size, MAX_IMAGE_SIZE))

    def save_image_to_cache(image: Image.Image) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(CACHE_DIR, f"{timestamp}.png")
        image.save(filename, format="PNG")
        return filename

    def delete_image_from_cache(filename: str) -> None:
        os.remove(filename)

    @app.get("/api/sd")
    async def api_txt2img(
        prompt: str,
        negative_prompt: str = "",
        steps: int = SD_DEFAULT_STEPS,
        guidance_scale: float = SD_DEFAULT_GUIDANCE_SCALE,
        nsfw=False,
    ):
        # Convert the prompt to lowercase for consistency
        prompt = prompt.lower()

        # Generate image for text-to-image request
        generated_image = text_to_image_pipe(
            prompt=("" if nsfw else "digital illustration:1.1, ")
            + prompt,
            negative_prompt=("child:1.1, teen:1.1, " if nsfw else "photo, realistic, nsfw, ")
            + "watermark, signature, "
            + negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=512,
            height=512,
        ).images[0]

        # Save the generated image to a temporary file
        temp_file = save_image_to_cache(generated_image)

        if nsfw:
            # try:
            response = FileResponse(path=temp_file, media_type="image/png")
            # delete_image_from_cache(temp_file)
            return response
        # finally:
        # Delete the temporary file
        # delete_image_from_cache(temp_file)
        else:
            # try:
            # Preprocess the image (replace this with your preprocessing logic)
            # Assuming nude_detector.censor returns the path of the processed image
            processed_image = nude_detector.censor(
                temp_file,
                [
                    "ANUS_EXPOSED",
                    "MALE_GENITALIA_EXPOSED",
                    "FEMALE_GENITALIA_EXPOSED",
                    "FEMALE_BREAST_EXPOSED",
                ],
            )
            delete_image_from_cache(temp_file)
            return FileResponse(path=processed_image, media_type="image/png")

        # finally:
        # Delete the temporary file
        # delete_image_from_cache(temp_file)
