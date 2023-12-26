from settings import SD_USE_HYPERTILE_VIDEO, TTS_VOICES_PATH
import gradio as gr
import logging
import io
import os
from diffusers.utils import export_to_video
from utils.file_utils import random_filename
from utils.gpu_utils import free_vram
from PIL import Image
from utils.gpu_utils import gpu_thread_lock

logging.basicConfig(level=logging.INFO)

settings = {
    "language": "en",
    "speed": 1,
    "temperature": 0.75,
    "voice": os.path.join(TTS_VOICES_PATH, "female1.wav"),
}


def launch_webui(args, prevent_thread_lock=False):
    if args is None or args.tts:
        from clients.TTSClient import TTSClient

        tts = TTSClient()
    else:
        tts = None

    async def chat(text: str, history: list[list], chunk_sentences=True):
        print(f"text={text}")
        print(f"chunk_sentences={chunk_sentences}")

        response = Exllama2Client().chat(
            text=text,
            messages=convert_gr_to_openai(history),
            chunk_sentences=chunk_sentences,
        )

        message = ""
        for chunk in response:
            if chunk_sentences:
                message += " " + chunk

                if (tts is not None) and grChatSpeak.value:
                    print("")
                    logging.info("\nGenerating speech...")
                    async with gpu_thread_lock:
                        free_vram("tts", TTSClient())
                        audio = tts.generate_speech(
                            chunk,
                            speed=settings["speed"],
                            temperature=settings["temperature"],
                            speaker_wav=settings["voice"],
                            language=settings["language"],
                        )
                    yield message.strip()
                    play_wav_from_bytes(audio)

            else:
                message += chunk
                yield message

    with gr.Blocks(title="monofy-ai", analytics_enabled=False).queue() as web_ui:
        if not args or args.llm:
            from clients.Exllama2Client import Exllama2Client

            from utils.chat_utils import convert_gr_to_openai

            with gr.Tab("Chat/TTS"):
                grChatSpeak = None

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            grChatSentences = gr.Checkbox(
                                value=True, label="Chunk full sentences"
                            )
                            grChatSpeak = gr.Checkbox(
                                value=tts is not None,
                                interactive=tts is not None,
                                label="Speak results",
                            )
                        gr.ChatInterface(
                            fn=chat, additional_inputs=[grChatSentences]
                        ).queue()

                    if tts:
                        import pygame

                        def play_wav_from_bytes(wav_bytes):
                            pygame.mixer.init()
                            sound = pygame.mixer.Sound(io.BytesIO(wav_bytes))
                            sound.play()

                            # Wait for the sound to finish playing
                            pygame.time.wait(int(sound.get_length() * 1000))

                        with gr.Column():

                            def set_language(value):
                                settings["language"] = value

                            def set_speed(value):
                                settings["speed"] = value

                            def set_temperature(value):
                                settings["temperature"] = value

                            def set_voice(value):
                                settings["voice"] = value

                            with gr.Column():
                                grText = gr.Textbox(
                                    "This is a test of natural speech.", label="Text"
                                )
                                tts_voice = gr.Textbox(
                                    "voices/female1.wav", label="Voice"
                                )
                                with gr.Row():
                                    tts_speed = gr.Number("1", label="Speed")
                                    tts_temperature = gr.Number(
                                        "0.75", label="Temperature"
                                    )

                                    tts_language = gr.Dropdown(
                                        [
                                            "en",
                                            "es",
                                            "fr",
                                            "de",
                                            "it",
                                            "pt",
                                            "pl",
                                            "tr",
                                            "ru",
                                            "nl",
                                            "cs",
                                            "ar",
                                            "zh-cn",
                                            "ja",
                                            "hu",
                                            "ko",
                                        ],
                                        label="Language",
                                        value=settings["language"],
                                    )
                                tts_language.change(set_language, inputs=[tts_language])
                                tts_speed.change(set_speed, inputs=[tts_speed])
                                tts_temperature.change(
                                    set_temperature, inputs=[tts_temperature]
                                )
                                tts_voice.change(set_voice, inputs=[tts_voice])
                                tts_button = gr.Button("Generate")
                                tts_output = gr.Audio(
                                    label="Audio Output",
                                    type="numpy",
                                    autoplay=True,
                                    format="wav",
                                    interactive=False,
                                    streaming=False,  # TODO
                                )

                            async def preview_speech(
                                text: str,
                                speed: int,
                                temperature: float,
                                voice: str,
                                language: str,
                            ):
                                # TODO stream to grAudio using generate_text_streaming
                                async with gpu_thread_lock:
                                    free_vram("tts", TTSClient())
                                    yield TTSClient().generate_speech(
                                        text,
                                        speed,
                                        temperature,
                                        voice,
                                        language,
                                    )

                            tts_button.click(
                                preview_speech,
                                inputs=[
                                    grText,
                                    tts_speed,
                                    tts_temperature,
                                    tts_voice,
                                    tts_language,
                                ],
                                outputs=[tts_output],
                            )

                        # Right half of the screen (Chat UI) - Only if args.llm is True

        if not args or args.sd:
            from clients.SDClient import SDClient
            from clients.AudioGenClient import AudioGenClient
            from clients.MusicGenClient import MusicGenClient
            from hyper_tile import split_attention

            t2i_vid_button: gr.Button = None

            async def generate_video(
                image_input,
                width: int,
                height: int,
                steps: int,
                fps: int,
                motion_bucket_id: int,
                noise: float,
            ):
                # Convert numpy array to PIL Image
                async with gpu_thread_lock:
                    free_vram("svd", SDClient()) # TODO VideoClient()
                    image = Image.fromarray(image_input).convert("RGB")
                    filename_noext = random_filename(None, True)
                    num_frames = 25

                    def do_gen():
                        video_frames = SDClient().video_pipeline(
                            image,
                            num_inference_steps=steps,
                            num_frames=num_frames,
                            motion_bucket_id=motion_bucket_id,
                            decode_chunk_size=num_frames,
                            width=width,
                            height=height,
                            noise_aug_strength=noise,
                        ).frames[0]
                        export_to_video(video_frames, f"{filename_noext}.mp4", fps=fps)
                        return f"{filename_noext}.mp4"

                    if SD_USE_HYPERTILE_VIDEO:
                        aspect_ratio = 1 if width == height else width / height
                        split_vae = split_attention(
                            SDClient().video_pipeline.vae,
                            tile_size=256,
                            aspect_ratio=aspect_ratio,
                        )
                        split_unet = split_attention(
                            SDClient().video_pipeline.unet,
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
                async with gpu_thread_lock:
                    free_vram("stable diffusion", SDClient())
                    result = SDClient().txt2img(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                    )
                    yield result.images[0], gr.Button(
                        label="Generate Video", interactive=True
                    )

            async def audiogen(prompt: str):
                filename_noext = random_filename(None, True)
                return AudioGenClient().generate(
                    prompt, file_path=filename_noext
                )

            async def musicgen(prompt: str):
                file_path = random_filename("wav", True)
                return MusicGenClient().generate(prompt, file_path=file_path)

            def disable_send_button():
                yield gr.Button(label="Generating...", interactive=False)

            with gr.Tab("Image/Video"):
                with gr.Row():
                    with gr.Column():
                        t2i_prompt = gr.TextArea(
                            "an advanced humanoid robot with human expression in a futuristic laboratory",
                            lines=4,
                            label="Prompt",
                        )
                        t2i_negative_prompt = gr.TextArea(
                            "", lines=4, label="Negative Prompt"
                        )
                        with gr.Row():
                            t2i_width = gr.Slider(
                                minimum=256,
                                maximum=2048,
                                value=512,
                                step=8,
                                interactive=True,
                                label="Width",
                            )
                            t2i_height = gr.Slider(
                                minimum=256,
                                maximum=2048,
                                value=512,
                                step=8,
                                interactive=True,
                                label="Height",
                            )
                        t2i_steps = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1,
                            interactive=True,
                            label="Steps",
                        )
                        t2i_guidance_scale = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=3,
                            step=0.1,
                            interactive=True,
                            label="Guidance",
                        )
                        t2i_button = gr.Button("Generate")
                    with gr.Column():
                        t2i_output = gr.Image(
                            None,
                            width=512,
                            height=512,
                            interactive=False,
                            label="Output",
                        )
                        with gr.Row():
                            i2v_width = gr.Number(
                                320, label="Width", precision=0, step=8
                            )
                            i2v_height = gr.Number(
                                320, label="Height", precision=0, step=8
                            )
                            i2v_fps = gr.Number(6, label="FPS", precision=0)
                            i2v_steps = gr.Number(10, label="Steps", precision=0)
                            i2v_motion = gr.Number(
                                15, label="Motion Bucket ID", precision=0
                            )
                            i2v_noise = gr.Number(
                                0.0,
                                label="Noise (also increases motion)",
                                precision=0,
                                step=0.01,
                            )

                        t2i_vid_button = gr.Button("Generate Video", interactive=False)

                        i2v_output = gr.Video(
                            None,
                            width=320,
                            height=320,
                            interactive=False,
                            label="Video",
                            format="mp4",
                            autoplay=True
                        )

                        t2i_vid_button.click(
                            generate_video,
                            inputs=[
                                t2i_output,
                                i2v_width,
                                i2v_height,
                                i2v_steps,
                                i2v_fps,
                                i2v_motion,
                                i2v_noise,
                            ],
                            outputs=[i2v_output],
                        )

                    t2i_button.click(disable_send_button, outputs=[t2i_vid_button])
                    t2i_button.click(
                        txt2img,
                        inputs=[
                            t2i_prompt,
                            t2i_negative_prompt,
                            t2i_width,
                            t2i_height,
                            t2i_steps,
                            t2i_guidance_scale,
                        ],
                        outputs=[t2i_output, t2i_vid_button],
                    )

            with gr.Tab("Audio/Music"):
                with gr.Row():
                    with gr.Column():
                        audiogen_prompt = gr.TextArea(
                            label="Audio description", lines=3
                        )
                        audiogen_button = gr.Button("Generate Audio")
                        audiogen_output = gr.Audio(interactive=False)
                        audiogen_button.click(
                            audiogen,
                            inputs=[audiogen_prompt],
                            outputs=[audiogen_output],
                        )
                    with gr.Column():
                        musicgen_prompt = gr.TextArea(
                            label="Music description", lines=3
                        )
                        musicgen_button = gr.Button("Generate Music")
                        musicgen_output = gr.Audio(interactive=False)
                        musicgen_button.click(
                            musicgen,
                            inputs=[musicgen_prompt],
                            outputs=[musicgen_output],
                        )

        web_ui.launch(
            prevent_thread_lock=prevent_thread_lock, inbrowser=args and not args.all
        )


if __name__ == "__main__":
    print("Loading webui in main thread.")
    launch_webui(True, True, True, False)
