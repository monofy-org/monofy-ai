import io
import logging
import random
import string
from urllib.parse import unquote
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from clients.musicgen.AudioGenClient import AudioGenClient
from clients.musicgen.MusicGenClient import MusicGenClient
from clients.sd.SDClient import SDClient
from clients.shape.ShapeClient import ShapeClient
from utils.gpu_utils import gpu_thread_lock
from utils.file_utils import delete_file, random_filename
from utils.image_utils import detect_objects
from diffusers.utils import load_image, export_to_video
from PIL import Image
from nudenet import NudeDetector
from settings import (
    DEVICE,
    SD_DEFAULT_STEPS,
    SD_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_WIDTH,
    SD_DEFAULT_HEIGHT,
    SD_USE_HYPERTILE,
    MEDIA_CACHE_DIR,
)
from utils.gpu_utils import free_vram
from utils.math_utils import limit
from utils.video_utils import double_frame_rate_with_interpolation
from hyper_tile import split_attention


def sd_api(app: FastAPI):
    nude_detector = NudeDetector()

    MAX_IMAGE_SIZE = (1024, 1024)
    MAX_FRAMES = 30

    def is_image_size_valid(image: Image.Image) -> bool:
        return all(dim <= size for dim, size in zip(image.size, MAX_IMAGE_SIZE))

    def save_image_to_cache(image: Image.Image) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(MEDIA_CACHE_DIR, f"{timestamp}.png")
        image.save(filename, format="PNG")
        return filename

    @app.get("/api/img2vid")
    async def img2vid(
        image_url: str,
        motion_bucket: int = 127,
        steps: int = 10,
        width: int = 512,
        height: int = 512,
        fps: int = 6,
        frames: int = 30,
        noise: float = 0,
        interpolate=3,
    ):
        with gpu_thread_lock:
            free_vram("svd")

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

            if width > 320 or height > 320:
                SDClient.instance.video_pipeline.enable_sequential_cpu_offload(0)
            else:
                SDClient.instance.video_pipeline.enable_model_cpu_offload(0)

            video_frames = SDClient.instance.video_pipeline(
                image,
                decode_chunk_size=frames,
                num_inference_steps=steps,
                generator=SDClient.instance.generator,
                num_frames=frames,
                width=width,
                height=height,
                motion_bucket_id=motion_bucket,
                noise_aug_strength=noise,
            ).frames[0]

            filename_noext = os.path.join(MEDIA_CACHE_DIR, random_filename())

            export_to_video(video_frames, f"{filename_noext}-0.mp4", fps=fps)

            interpolate = limit(interpolate, 0, 3)

            for i in range(0, interpolate):
                double_frame_rate_with_interpolation(
                    f"{filename_noext}-{i}.mp4", f"{filename_noext}-{i+1}.mp4"
                )

            print(f"Returning generated-{interpolate}.mp4...")
            return FileResponse(
                f"{filename_noext}-{interpolate}.mp4", media_type="video/mp4"
            )

    @app.get("/api/txt2img")
    async def txt2img(
        background_tasks: BackgroundTasks,
        prompt: str,
        negative_prompt: str = "",
        steps: int = SD_DEFAULT_STEPS,
        guidance_scale: float = SD_DEFAULT_GUIDANCE_SCALE,
        width: int = SD_DEFAULT_WIDTH,
        height: int = SD_DEFAULT_HEIGHT,
        nsfw: bool = False,
        upscale: bool = False,
    ):
        with gpu_thread_lock:
            free_vram("stable diffusion")

            # Convert the prompt to lowercase for consistency
            prompt = prompt.lower()

            def do_gen():
                # Generate image for text-to-image request
                return SDClient.instance.txt2img(
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

            temp_file = None

            if SD_USE_HYPERTILE and (width > 512 or height > 512):
                with split_attention(
                    SDClient.instance.image_pipeline.vae,
                    tile_size=256,
                    aspect_ratio=1 if width == height else width / height,
                ):
                    with split_attention(
                        SDClient.instance.image_pipeline.unet,
                        tile_size=256,
                        aspect_ratio=1 if width == height else width / height,
                    ):
                        generated_image = do_gen()

            else:
                generated_image = do_gen()

            if upscale:
                scale_factor = 2.5 if upscale else 1

                if SD_USE_HYPERTILE and (
                    width * scale_factor > 512 or height * scale_factor > 512
                ):
                    with split_attention(
                        SDClient.instance.image_pipeline.vae,
                        tile_size=256,
                        aspect_ratio=1 if width == height else width / height,
                    ):
                        with split_attention(
                            SDClient.instance.image_pipeline.unet,
                            tile_size=256,
                            aspect_ratio=1 if width == height else width / height,
                        ):
                            generated_image = SDClient.instance.upscale(
                                generated_image,
                                width,
                                height,
                                prompt,
                                negative_prompt,
                                steps,
                            )
                    temp_file = save_image_to_cache(generated_image)

                else:
                    generated_image = SDClient.instance.upscale(
                        generated_image,
                        width,
                        height,
                        prompt,
                        negative_prompt,
                        steps,
                    )
                    temp_file = save_image_to_cache(generated_image)

            if nsfw:
                background_tasks.add_task(delete_file, temp_file)
                return FileResponse(path=temp_file, media_type="image/png")
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
                delete_file(temp_file)
                background_tasks.add_task(delete_file, processed_image)
                return FileResponse(path=processed_image, media_type="image/png")

    @app.get("/api/shape")
    async def shape_api(
        background_tasks: BackgroundTasks,
        prompt: str,
        guidance_scale: float = 15.0,
        format: str = "gif",
    ):
        try:
            with gpu_thread_lock:
                filename_noext = random_filename()
                file_path = os.path.join(".cache", f"{filename_noext}.gif")
                ShapeClient.instance.generate(
                    prompt, file_path, guidance_scale=guidance_scale, format=format
                )
                background_tasks.add_task(delete_file, file_path)
                return FileResponse(os.path.abspath(file_path), media_type="image/gif")
        except Exception as e:
            logging.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/detect")
    async def object_detection(background_tasks: BackgroundTasks, image_url: str):
        try:
            with gpu_thread_lock:
                result_image = detect_objects(image_url, 0.8)
                img_byte_array = io.BytesIO()
                result_image.save(img_byte_array, format="PNG")
                return StreamingResponse(
                    io.BytesIO(img_byte_array.getvalue()), media_type="image/png"
                )
        except Exception as e:
            logging.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/audiogen")
    async def audiogen(
        background_tasks: BackgroundTasks,
        prompt: str,
        duration: int = 3,
        temperature: float = 1.0,
    ):
        try:
            with gpu_thread_lock:
                free_vram("audiogen")
                random_letters = "".join(
                    random.choice(string.ascii_letters) for _ in range(10)
                )
                file_path_noext = os.path.join(MEDIA_CACHE_DIR, f"{random_letters}")
                print(file_path_noext)
                AudioGenClient.instance.generate(
                    prompt, file_path_noext, duration=duration
                )
                file_path = f"{file_path_noext}.wav"
                background_tasks.add_task(delete_file, file_path)
                return FileResponse(os.path.abspath(file_path), media_type="audio/wav")
        except Exception as e:
            logging.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/musicgen")
    async def musicgen(
        background_tasks: BackgroundTasks,
        prompt: str,
        duration: int = 5,
        temperature: float = 1.0,
        cfg_coeff: float = 3.0,
    ):
        with gpu_thread_lock:
            free_vram("musicgen")
            try:
                filename_noext = random_filename()
                file_path_noext = os.path.join(MEDIA_CACHE_DIR, f"{filename_noext}")
                print(file_path_noext)
                MusicGenClient.instance.generate(
                    prompt,
                    file_path_noext,
                    duration=duration,
                    temperature=temperature,
                    cfg_coeff=cfg_coeff,
                )
                file_path = f"{file_path_noext}.wav"
                background_tasks.add_task(delete_file, file_path)
                return FileResponse(os.path.abspath(file_path), media_type="audio/wav")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
