import io
import logging
import os
import modules
import gradio as gr
from settings import SD_USE_HYPERTILE_VIDEO, SD_USE_SDXL, TTS_VOICES_PATH
from submodules.HyperTile.hyper_tile.hyper_tile import split_attention
from utils.chat_utils import convert_gr_to_openai
from utils.file_utils import random_filename
from utils.gpu_utils import load_gpu_task, gpu_thread_lock
from diffusers.utils import export_to_video
from PIL import Image

settings = {
    "language": "en",
    "speed": 1,
    "temperature": 0.75,
    "voice": os.path.join(TTS_VOICES_PATH, "female1.wav"),
}


def set_language(value):
    settings["language"] = value


def set_speed(value):
    settings["speed"] = value


def set_temperature(value):
    settings["temperature"] = value


def set_voice(value):
    settings["voice"] = value


def play_wav_from_bytes(wav_bytes):

    import pygame

    pygame.mixer.init()
    sound = pygame.mixer.Sound(io.BytesIO(wav_bytes))
    sound.play()

    # Wait for the sound to finish playing
    pygame.time.wait(int(sound.get_length() * 1000))


async def chat(text: str, history: list[list], speak_results: bool, chunk_sentences):
    from clients import TTSClient, Exllama2Client

    print(f"text={text}")
    print(f"chunk_sentences={chunk_sentences}")

    response = Exllama2Client.chat(
        text=text,
        messages=convert_gr_to_openai(history),
    )

    message = ""
    for chunk in response:
        if not speak_results:
            yield chunk
        message += chunk

    if speak_results:
        logging.info("\nGenerating speech...")
        async with gpu_thread_lock:
            load_gpu_task("tts", TTSClient)

            audio = TTSClient.generate_speech(
                message,
                speed=settings["speed"],
                temperature=settings["temperature"],
                speaker_wav=settings["voice"],
                language=settings["language"],
            )
            play_wav_from_bytes(audio)
            yield message


async def preview_speech(
    text: str,
    speed: int,
    temperature: float,
    voice: str,
    language: str,
):
    from clients import TTSClient

    # TODO stream to grAudio using generate_text_streaming
    async with gpu_thread_lock:
        load_gpu_task("tts", TTSClient)

        yield TTSClient.generate_speech(
            text,
            speed,
            temperature,
            voice,
            language,
        )


async def generate_video(
    image_input,
    width: int,
    height: int,
    steps: int,
    fps: int,
    motion_bucket_id: int,
    noise: float,
    interpolate: int,
):
    from clients import SDClient

    # Convert numpy array to PIL Image
    async with gpu_thread_lock:
        load_gpu_task("img2vid", SDClient)  # TODO VideoClient
        SDClient.init_img2vid()
        image = Image.fromarray(image_input).convert("RGB")
        filename_noext = random_filename()
        num_frames = 50
        decode_chunk_size = 25

        def do_gen():
            video_frames = SDClient.pipelines["img2vid"](
                image,
                num_inference_steps=steps,
                num_frames=num_frames,
                motion_bucket_id=motion_bucket_id,
                decode_chunk_size=decode_chunk_size,
                width=width,
                height=height,
                noise_aug_strength=noise,
            ).frames[0]

            if interpolate > 1:
                video_frames = modules.rife.interpolate(
                    video_frames,
                    count=interpolate,
                    scale=1,
                    pad=1,
                    change=0,
                )
                export_to_video(
                    video_frames,
                    f"{filename_noext}.mp4",
                    fps=fps * interpolate,
                )

            else:
                export_to_video(video_frames, f"{filename_noext}.mp4", fps=fps)

            return f"{filename_noext}.mp4"

        if SD_USE_HYPERTILE_VIDEO:
            aspect_ratio = 1 if width == height else width / height
            split_vae = split_attention(
                SDClient.pipelines["img2vid"].vae,
                tile_size=256,
                aspect_ratio=aspect_ratio,
            )
            split_unet = split_attention(
                SDClient.pipelines["img2vid"].unet,
                tile_size=256,
                aspect_ratio=aspect_ratio,
            )
            with split_vae:
                with split_unet:
                    yield do_gen()

        else:
            yield do_gen()


async def txt2img(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
):
    from clients import SDClient

    async with gpu_thread_lock:
        load_gpu_task("sdxl" if SD_USE_SDXL else "stable diffusion", SDClient)
        SDClient.load_model()
        result = SDClient.pipelines["txt2img"](
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
    yield result.images[0], gr.Button(label="Generate Video", interactive=True)


async def audiogen(prompt: str, duration: float, temperature: float):
    from clients import AudioGenClient

    filename_noext = random_filename()
    return AudioGenClient.generate(
        prompt,
        file_path=filename_noext,
        duration=duration,
        temperature=temperature,
    )


async def musicgen(prompt: str, duration: float, temperature: float):
    from clients import MusicGenClient

    filename_noext = random_filename()
    return MusicGenClient.generate(
        prompt,
        output_path=filename_noext,
        duration=duration,
        temperature=temperature,
    )

async def shape_generate(prompt: str, steps: int, guidance: float):
    from clients import ShapeClient

    async with gpu_thread_lock:
        load_gpu_task("shap-e", ShapeClient)
        filename_noext = random_filename()
        file_path = ShapeClient.generate(
            prompt,
            steps=steps,
            guidance_scale=guidance,
            file_path=filename_noext,
            format="glb",
        )
        print(file_path)
        yield file_path

def disable_send_button():
    yield gr.Button(label="Generating...", interactive=False)
