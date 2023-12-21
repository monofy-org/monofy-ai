import logging
import io

from clients.llm.Exllama2Client import Exllama2Client
from clients.llm.chat_utils import convert_gr_to_openai
from clients.musicgen.AudioGenClient import AudioGenClient
from clients.musicgen.MusicGenClient import MusicGenClient
from clients.sd.SDClient import SDClient
from settings import LOG_LEVEL
from ttsclient import TTSClient
from utils.gpu_utils import free_vram
import gradio as gr

logging.basicConfig(level=LOG_LEVEL)

settings = {
    "language": "en",
    "speed": 1,
    "temperature": 0.75,
    "voice": "voices/female1.wav",
}


def launch_webui(use_llm=False, use_tts=False, use_sd=False, prevent_thread_lock=False):
    tts_client: TTSClient = None
    exllamav2_client: Exllama2Client = None

    if use_tts:
        tts_client = TTSClient.instance

    if use_llm:
        exllamav2_client = Exllama2Client.instance

    with gr.Blocks(title="monofy-ai", analytics_enabled=False).queue() as web_ui:
        if use_llm:
            with gr.Tab("Chat"):
                grChatSpeak = None

                async def chat(text: str, history: list[list], chunk_sentences=True):
                    print(f"text={text}")
                    print(f"chunk_sentences={chunk_sentences}")
                    free_vram("exllamav2", exllamav2_client)
                    response = exllamav2_client.chat(
                        text=text,
                        messages=convert_gr_to_openai(history),
                        chunk_sentences=chunk_sentences,
                    )

                    message = ""
                    for chunk in response:
                        if chunk_sentences:
                            message += " " + chunk

                            if tts_client and grChatSpeak.value:
                                print("")
                                logging.info("\nGenerating speech...")

                                audio = tts_client.generate_speech(
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

                with gr.Column():
                    with gr.Row():
                        grChatSentences = gr.Checkbox(
                            value=True, label="Chunk full sentences"
                        )
                        grChatSpeak = gr.Checkbox(
                            value=use_tts,
                            interactive=use_tts,
                            label="Speak results",
                        )
                    gr.ChatInterface(
                        fn=chat, additional_inputs=[grChatSentences]
                    ).queue()

        if use_tts:
            with gr.Tab("Speech"):
                import simpleaudio as sa

                def play_wav_from_bytes(wav_bytes):
                    wave_obj = sa.WaveObject.from_wave_file(io.BytesIO(wav_bytes))
                    play_obj = wave_obj.play()

                    # Wait for the sound to finish playing
                    play_obj.wait_done()

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
                    tts_voice = gr.Textbox("voices/female1.wav", label="Voice")
                    with gr.Row():
                        tts_speed = gr.Number("1", label="Speed")
                        tts_temperature = gr.Number("0.75", label="Temperature")

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
                    tts_temperature.change(set_temperature, inputs=[tts_temperature])
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
                        yield tts_client.generate_speech(
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

                # Right half of the screen (Chat UI) - Only if use_llm is True

        if use_sd:
            sd_client = SDClient.instance
            audiogen_client = AudioGenClient.instance
            musicgen_client = MusicGenClient.instance

            t2i_send_button: gr.Button = None

            def send_to_video(fromImage):
                yield fromImage

            async def txt2img(
                prompt: str,
                negative_prompt: str,
                width: int,
                height: int,
                num_inference_steps: int,
                guidance_scale: float,
            ):
                free_vram("stable diffusion")
                result = sd_client.txt2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                )
                yield result.images[0], t2i_send_button.update(interactive=True)

            async def img2vid(image):
                free_vram("svd")
                result = sd_client.video_pipeline(image)
                yield result.images[0]

            async def audiogen(prompt: str):
                free_vram("audiogen")
                return audiogen_client.generate(prompt)

            async def musicgen(prompt: str):
                free_vram("musicgen")
                return musicgen_client.generate(prompt)

            def disable_send_button():
                yield t2i_send_button.update(interactive=False)

            with gr.Tab("Image"):
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
                            step=1,
                            interactive=True,
                            label="Guidance Scale",
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
                        t2i_send_button = gr.Button("Send to Video", interactive=False)

                    t2i_button.click(disable_send_button, outputs=[t2i_send_button])
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
                        outputs=[t2i_output, t2i_send_button],
                    )

            with gr.Tab("Video"):
                with gr.Row():
                    with gr.Column():
                        i2v_input = gr.Image(
                            None, width=512, height=512, label="Input Image"
                        )
                        gr.Button("Send to Video", interactive=False).click(
                            send_to_video, inputs=[t2i_output], outputs=[i2v_input]
                        )
                    with gr.Column():
                        gr.PlayableVideo(
                            None,
                            width=320,
                            height=320,
                            interactive=False,
                            label="Output",
                        )

            with gr.Tab("Audio"):
                with gr.Row():
                    with gr.Column():
                        audiogen_prompt = gr.TextArea(
                            label="Sound description", lines=3
                        )
                        audiogen_button = gr.Button("Generate SFX")
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

        web_ui.launch(prevent_thread_lock=prevent_thread_lock, inbrowser=True)


if __name__ == "__main__":
    print("Loading webui in main thread.")
    launch_webui(True, True, True, False)
