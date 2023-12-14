import torch
import os
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import FileResponse
from clients.sdclient import SDClient
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
from utils.video_utils import double_frame_rate_with_interpolation

nude_detector = NudeDetector()

SD_ALLOW_IMAGES = True


def sd_api(app: FastAPI):
    device = autodetect_device()
    print(f"Stable Diffusion using device: {device}")

    client = SDClient()
    #client.video_pipeline.enable_model_cpu_offload(0)

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
    async def api_img2vid(
        image_url: str,
        motion_bucket: int = 3,
        steps: int = 10,
        width: int = 320,
        height: int = 320,
    ):
        image = load_image(image_url)
        # s = image.width if image.width < image.height else image.height
        # image = image.crop((0, 0, s, s))
        if image.width < image.height:
            s = image.width
            offset = (image.height - image.width) // 2
            image = image.crop((0, offset, s, image.height - offset))
        else:
            s = image.height
            offset = (image.width - image.height) // 2
            image = image.crop((offset, 0, image.width - offset, s))
        image = image.resize((1024, 1024))

        client.video_pipeline.to(device)
        # client.image_to_video_pipe.unet.enable_forward_chunking()
        
        frames = client.video_pipeline(
            image,
            decode_chunk_size=12,
            num_inference_steps=steps,
            generator=client.generator,
            num_frames=24,
            width=width,
            height=height,
            motion_bucket_id=motion_bucket,
        ).frames[0]

        client.video_pipeline.to("cpu")
        torch.cuda.empty_cache()
        
        export_to_video(frames, "generated.mp4", fps=6)        

        double_frame_rate_with_interpolation(
            "generated.mp4", "generated-interpolated.mp4"
        )

        return FileResponse("generated-interpolated.mp4", media_type="video/mp4")

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
            response = FileResponse(path=processed_image, media_type="image/png")
            delete_image_from_cache(processed_image)
            return response
