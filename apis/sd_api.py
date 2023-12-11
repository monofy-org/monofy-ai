import torch
import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import FileResponse
from apis.sdclient import SDClient
from utils.torch_utils import autodetect_device
from diffusers.utils import load_image, export_to_video
from PIL import Image
from nudenet import NudeDetector
from settings import (
    SD_DEFAULT_STEPS,
    SD_DEFAULT_GUIDANCE_SCALE,
    SD_IMAGE_WIDTH,
    SD_IMAGE_HEIGHT,
)

nude_detector = NudeDetector()

SD_ALLOW_IMAGES = True


def sd_api(app: FastAPI):
    device = autodetect_device()
    print(f"Stable Diffusion using device: {device}")

    client = SDClient()

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

    @app.get("/api/img2vid")
    async def api_img2vid(image_url: str):
        # Load the conditioning image
        image = load_image(image_url)
        # image = image.resize((1024, 576))

        # image_to_video_pipe.enable_model_cpu_offload()
        client.video_pipeline.to(device)
        # image_to_video_pipe.unet.enable_forward_chunking()
        frames = client.video_pipeline(
            image,
            decode_chunk_size=8,
            num_inference_steps=15,
            generator=client.generator,
            num_frames=48,
            width=320,
            height=512,
            motion_bucket_id=4,
        ).frames[0]

        vid = export_to_video(frames, "generated.mp4", fps=6)

        return FileResponse(vid)

    @app.get("/api/txt2img")
    async def api_txt2img(
        prompt: str,
        negative_prompt: str = "",
        steps: int = SD_DEFAULT_STEPS,
        guidance_scale: float = SD_DEFAULT_GUIDANCE_SCALE,
        nsfw=False,
    ):
        # Convert the prompt to lowercase for consistency
        prompt = prompt.lower()

        client.image_pipeline.to(device)

        # Generate image for text-to-image request
        generated_image = client.txt2img(
            prompt=("" if nsfw else "digital illustration:1.1, ") + prompt,
            negative_prompt=(
                "child:1.1, teen:1.1, " if nsfw else "photo, realistic, nsfw, "
            )
            + "watermark, signature, "
            + negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=SD_IMAGE_WIDTH,
            height=SD_IMAGE_HEIGHT,
        ).images[0]

        client.image_pipeline.to("cpu")
        torch.cuda.empty_cache()

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
