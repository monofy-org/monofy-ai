import io
import logging
import time
from urllib.parse import unquote
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
import torch
from utils.gpu_utils import get_seed, gpu_thread_lock
from utils.file_utils import delete_file, random_filename
from utils.image_utils import detect_objects
from diffusers.utils import load_image, export_to_video
from PIL import Image
from nudenet import NudeDetector
from settings import (
    SD_DEFAULT_STEPS,
    SD_DEFAULT_GUIDANCE_SCALE,
    SD_DEFAULT_WIDTH,
    SD_DEFAULT_HEIGHT,
    SD_USE_HYPERTILE,
    MEDIA_CACHE_DIR,
    SD_USE_HYPERTILE_VIDEO,
)
from utils.gpu_utils import load_gpu_task
from utils.math_utils import limit
from utils.misc_utils import print_completion_time
from utils.video_utils import double_frame_rate_with_interpolation
from hyper_tile import split_attention


def diffusers_api(app: FastAPI):
    nude_detector = NudeDetector()

    MAX_IMAGE_SIZE = (1024, 1024)
    MAX_FRAMES = 25

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
        frames: int = MAX_FRAMES,
        noise: float = 0,
        interpolate=3,
        seed=-1,
    ):
        async with gpu_thread_lock:
            start_time = time.time()
            from clients import SDClient

            load_gpu_task("svd", SDClient)  # TODO: VideoClient

            url = unquote(image_url)
            image = load_image(url)

            aspect_ratio = width / height
            
            if aspect_ratio < 1: # portrait
                image = image.crop((0, 0, image.height * aspect_ratio, image.height))                            
            elif aspect_ratio > 1: # landscape
                image = image.crop((0, 0, image.width, image.width / aspect_ratio))
            else: # square
                dim = min(image.width, image.height)
                image = image.crop((0, 0, dim, dim))

            image = image.resize((width, height), Image.Resampling.BICUBIC)

            if frames > MAX_FRAMES:
                frames = MAX_FRAMES

            def process_and_get_response(frames, interpolate):
                filename_noext = random_filename(None, True)

                export_to_video(frames, f"{filename_noext}-0.mp4", fps=fps)

                interpolate = limit(interpolate, 0, 3)

                for i in range(0, interpolate):
                    double_frame_rate_with_interpolation(
                        f"{filename_noext}-{i}.mp4", f"{filename_noext}-{i+1}.mp4"
                    )

                print(f"Returning generated-{interpolate}.mp4...")
                return FileResponse(
                    f"{filename_noext}-{interpolate}.mp4", media_type="video/mp4"
                )

            def gen():
                video_frames = SDClient.video_pipeline(
                    image,
                    decode_chunk_size=frames,
                    num_inference_steps=steps,
                    generator=get_seed(seed),
                    num_frames=frames,
                    width=width,
                    height=height,
                    motion_bucket_id=motion_bucket,
                    noise_aug_strength=noise,
                ).frames[0]

                return process_and_get_response(video_frames, interpolate)

            if SD_USE_HYPERTILE_VIDEO:
                aspect_ratio = 1 if width == height else width / height
                split_vae = split_attention(
                    SDClient.video_pipeline.vae,
                    tile_size=256,
                    aspect_ratio=aspect_ratio,
                )
                split_unet = split_attention(
                    SDClient.video_pipeline.unet,
                    tile_size=256,
                    aspect_ratio=aspect_ratio,
                )
                with split_vae:
                    with split_unet:
                        result = gen()
                        print_completion_time(start_time)
                        return result

            else:
                result = gen()
                print_completion_time(start_time)
                return result

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
        upscale: float = 0,
        upscale_strength: float = 0.6,
        canny: bool = False,
        widen_coef: float = 0,
        seed: int = -1,
    ):
        async with gpu_thread_lock:
            from clients import SDClient

            load_gpu_task("stable diffusion", SDClient)
            # Convert the prompt to lowercase for consistency
            prompt = prompt.lower()

            def do_gen():
                generator = get_seed(seed)
                generated_image = SDClient.txt2img(
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
                    generator=generator,
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
                    use_canny=canny,
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
                temp_file = save_image_to_cache(image)

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

            if SD_USE_HYPERTILE:
                split_vae = split_attention(
                    SDClient.vae,
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

                        return process_and_respond(generated_image)

            else:
                generated_image = do_gen()
                if upscale >= 1:
                    generated_image = do_upscale(generated_image)

                return process_and_respond(generated_image)

    @app.get("/api/shape")
    async def shape_api(
        background_tasks: BackgroundTasks,
        prompt: str,
        guidance_scale: float = 15.0,
        format: str = "gif",
    ):
        try:
            async with gpu_thread_lock:
                filename_noext = random_filename()
                file_path = os.path.join(".cache", f"{filename_noext}.gif")
                from clients import ShapeClient

                ShapeClient.generate(
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
            async with gpu_thread_lock:
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
            from clients import AudioGenClient

            async with gpu_thread_lock:
                file_path_noext = random_filename(None, True)
                file_path = AudioGenClient.generate(
                    prompt, file_path_noext, duration=duration, temperature=temperature
                )
                background_tasks.add_task(delete_file, file_path)
                return FileResponse(os.path.abspath(file_path), media_type="audio/wav")
        except Exception as e:
            logging.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/musicgen")
    async def musicgen(
        background_tasks: BackgroundTasks,
        prompt: str,
        duration: int = 8,
        temperature: float = 1.0,
        cfg_coef: float = 3.0,
    ):
        async with gpu_thread_lock:
            try:
                from clients import MusicGenClient

            
                file_path_noext = random_filename(None, True)
                file_path = MusicGenClient.generate(
                    prompt,
                    file_path_noext,
                    duration=duration,
                    temperature=temperature,
                    cfg_coef=cfg_coef,
                )
                background_tasks.add_task(delete_file, file_path)
                return FileResponse(
                    os.path.abspath(file_path), media_type="audio/wav"
                )
            except Exception as e:
                logging.error(e)
                raise HTTPException(status_code=500, detail=str(e))
