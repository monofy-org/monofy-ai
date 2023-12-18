import io
import logging
import random
import string
from urllib.parse import unquote
import threading
import torch
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from clients.musicgen.AudioGenClient import AudioGenClient
from clients.musicgen.MusicGenClient import MusicGenClient
from clients.sd.SDClient import SDClient
from utils.image_utils import detect_objects
from diffusers.utils import load_image, export_to_video
from PIL import Image
from nudenet import NudeDetector
from settings import (
    SD_DEFAULT_STEPS,
    SD_DEFAULT_GUIDANCE_SCALE,
    SD_IMAGE_WIDTH,
    SD_IMAGE_HEIGHT,
    MEDIA_CACHE_DIR,
)
from utils.video_utils import double_frame_rate_with_interpolation


def sd_api(app: FastAPI):
    thread_lock = threading.Lock()
    sd_client = SDClient()
    nude_detector = NudeDetector()
    audiogen_client = AudioGenClient()
    musicgen_client = MusicGenClient()

    MAX_IMAGE_SIZE = (1024, 1024)
    MAX_FRAMES = 30

    def is_image_size_valid(image: Image.Image) -> bool:
        return all(dim <= size for dim, size in zip(image.size, MAX_IMAGE_SIZE))

    def save_image_to_cache(image: Image.Image) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(MEDIA_CACHE_DIR, f"{timestamp}.png")
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
        fps: int = 6,
        frames: int = 36,
        noise: float = 0,
    ):
        with thread_lock:
            url = unquote(image_url)
            image = load_image(url)
            if image.width < image.height:
                s = image.width
                offset = (image.height - image.width) // 2
                image = image.crop((0, offset, s, image.height - offset))
            else:
                s = image.height
                offset = (image.width - image.height) // 2
                image = image.crop((offset, 0, image.width - offset, s))
            image = image.resize((1024, 1024))

            if frames > MAX_FRAMES:
                frames = MAX_FRAMES

            video_frames = sd_client.video_pipeline(
                image,
                decode_chunk_size=frames,
                num_inference_steps=steps,
                generator=sd_client.generator,
                num_frames=frames,
                width=width,
                height=height,
                motion_bucket_id=motion_bucket,
                noise_aug_strength=noise,
            ).frames[0]

            torch.cuda.empty_cache()

            export_to_video(video_frames, "generated.mp4", fps=fps)

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
        width: int = SD_IMAGE_WIDTH,
        height: int = SD_IMAGE_HEIGHT,
        nsfw: bool = False,
        upscale: bool = False,
    ):
        with thread_lock:
            # Convert the prompt to lowercase for consistency
            prompt = prompt.lower()

            # Generate image for text-to-image request
            generated_image = sd_client.txt2img(
                prompt=("" if nsfw else "digital illustration:1.1, ") + prompt,
                negative_prompt=(
                    "child:1.1, teen:1.1, deformed, extra limbs, extra fingers"
                    if nsfw
                    else "photo, realistic, nsfw, "
                )
                + "watermark, signature, "
                + negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]

            if upscale:
                generated_image = generated_image.resize(
                    (int(width * 1.25 * 2), int(height * 1.25 * 2)),
                    Image.Resampling.NEAREST,
                )
                generated_image = sd_client.img2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=generated_image,
                    num_inference_steps=steps,
                    strength=1,
                ).images[0]

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
                # delete_image_from_cache(processed_image)
                return response

    @app.get("/api/detect")
    async def detect_objects_api(image_url: str):
        try:
            result_image = detect_objects(image_url, 0.8)
            img_byte_array = io.BytesIO()
            result_image.save(img_byte_array, format="PNG")
            return StreamingResponse(
                io.BytesIO(img_byte_array.getvalue()), media_type="image/png"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/audiogen")
    async def audiogen_api(prompt: str, duration: int = 3):
        with thread_lock:
            try:
                random_letters = "".join(
                    random.choice(string.ascii_letters) for _ in range(10)
                )
                file_path_noext = os.path.join(MEDIA_CACHE_DIR, f"{random_letters}")
                print(file_path_noext)
                audiogen_client.generate(prompt, file_path_noext, duration=duration)
                file_path = f"{file_path_noext}.wav"
                response = FileResponse(
                    os.path.abspath(file_path), media_type="audio/wav"
                )
                # os.remove(file_path)
                return response
            except Exception as e:
                logging.error(e)
                raise HTTPException(status_code=500, detail=str(e))
            
    @app.get("/api/musicgen")
    async def musicgen_api(prompt: str, duration: int = 5):
        with thread_lock:
            try:
                random_letters = "".join(
                    random.choice(string.ascii_letters) for _ in range(10)
                )
                file_path_noext = os.path.join(MEDIA_CACHE_DIR, f"{random_letters}")
                print(file_path_noext)
                musicgen_client.generate(prompt, file_path_noext, duration=duration)
                file_path = f"{file_path_noext}.wav"
                response = FileResponse(
                    os.path.abspath(file_path), media_type="audio/wav"
                )
                # os.remove(file_path)
                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
