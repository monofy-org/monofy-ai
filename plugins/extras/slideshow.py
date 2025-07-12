import base64
import gc
import os
import shutil
import subprocess
import uuid

from fastapi.responses import FileResponse
from ffmpy import FFmpeg
from PIL import Image

from classes.requests import Txt2ImgRequest
from modules import plugins
from modules.plugins import check_low_vram, release_plugin, use_plugin
from plugins.stable_diffusion import StableDiffusionPlugin
from settings import CACHE_PATH
from utils.file_utils import random_filename


def save_image_to_file(img, path):
    if isinstance(img, Image.Image):
        img.save(path)
    elif isinstance(img, str):  # base64
        img_data = base64.b64decode(img.split(",")[-1])
        with open(path, "wb") as f:
            f.write(img_data)
    else:
        raise ValueError(
            "Unsupported image format. Must be PIL.Image or base64 string."
        )


def get_audio_duration(audio_path):
    """Get duration of audio file using torchaudio."""
    try:
        import torchaudio
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {e}")


def create_slideshow(images, output_path, length=None, audio=None):
    if not images:
        raise ValueError("Image list is empty.")

    num_images = len(images)

    # Determine total length
    if not length:
        if audio:
            length = get_audio_duration(audio)
        else:
            length = 60  # seconds

    duration_per_image = length / num_images
    fade_duration = 1  # seconds

    temp_dir = os.path.join(CACHE_PATH, f"slideshow_tmp_{uuid.uuid4().hex}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Save images to temp files
        image_paths = []
        for i, img in enumerate(images):            
            path = os.path.join(temp_dir, f"frame_{i:03d}.png")
            save_image_to_file(img, path)
            print(path)
            image_paths.append(path)

        # Get image dimensions from first image
        with Image.open(image_paths[0]) as im:
            width, height = im.size
        enlarged_width = int(width * 1.1)

        # Prepare FFmpeg input args
        input_args = []
        for path in image_paths:
            input_args.extend(
                [
                    "-loop",
                    "1",
                    "-t",
                    str(duration_per_image + fade_duration),
                    "-i",
                    path,
                ]
            )
        if audio:
            input_args.extend(["-i", audio])

        # Create filter complex
        filter_lines = []
        streams = []

        for i in range(num_images):
            # Scale while preserving aspect ratio, then crop from center
            # Use -1 to maintain aspect ratio, scale to fit the larger dimension
            filter_lines.append(
                f"[{i}:v]scale={enlarged_width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}[v{i}]"
            )
            streams.append(f"[v{i}]")

        # Build xfade chain properly
        if num_images == 1:
            final_output = streams[0]
        else:
            last = streams[0]
            for i in range(1, num_images):
                # Start the fade at the end of the previous image's display time
                offset = i * duration_per_image - fade_duration
                filter_lines.append(
                    f"{last}{streams[i]}xfade=transition=fade:duration={fade_duration}:offset={offset}[vxf{i}]"
                )
                last = f"[vxf{i}]"
            final_output = last

        output_args = [
            "-filter_complex",
            ";".join(filter_lines),
            "-map",
            final_output,
        ]

        if audio:
            output_args += ["-map", f"{num_images}:a", "-shortest"]

        output_args += ["-y"]

        ff = FFmpeg(inputs={}, outputs={output_path: input_args + output_args})

        ff.run()
        
        gc.collect()
        
        return output_path

    finally:
        # Clean up temporary directory
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        pass


@plugins.router.post("/txt2img/slideshow")
async def txt2img_slideshow(req: Txt2ImgRequest):
    plugin: StableDiffusionPlugin = None
    output_path = None
    req.num_images_per_prompt = 8
    try:
        plugin = await use_plugin(StableDiffusionPlugin)
        response = await plugin.generate(req)

        output_path = random_filename("mp4")
        slideshow = create_slideshow(response["images"], output_path)

        return FileResponse(slideshow, media_type="video/mp4")
    except Exception as e:
        raise RuntimeError(f"Failed to generate slideshow: {e}")
    finally:
        if plugin:
            release_plugin(StableDiffusionPlugin)
        # if output_path and os.path.exists(output_path):
        #     os.remove(output_path)