from settings import TTS_VOICES_PATH
from utils.startup_args import startup_args
import gradio as gr
import os
from utils.file_utils import random_filename
from utils.gpu_utils import load_gpu_task
from utils.gpu_utils import gpu_thread_lock
from utils.webui_functions import (
    settings,
    set_language,
    set_speed,
    set_temperature,
    set_voice,
    preview_speech,
    chat,
    generate_video,
    txt2img,
    audiogen,
    musicgen,
    disable_send_button,
)



def launch_webui(args, prevent_thread_lock=False):
    if args is None or args.tts:
        tts = True
    else:
        tts = False

    with gr.Blocks(title="monofy-ai", analytics_enabled=False) as web_ui:
        if not args or args.llm:
            with gr.Tab("Chat/TTS"):
                speech_checkbox = None

                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            speech_checkbox = gr.Checkbox(
                                value=tts is not None,
                                interactive=tts is not None,
                                label="Speak results",
                            )
                            check_sentences_checkbox = gr.Checkbox(
                                value=True,
                                label="Chunk sentences",
                                visible=False,  # TODO
                            )

                        grChat = gr.ChatInterface(
                            fn=chat,
                            additional_inputs=[
                                speech_checkbox,
                                check_sentences_checkbox,
                            ],
                        )
                        grChat.queue()

                    if tts:
                        with gr.Column():

                            with gr.Column():
                                grText = gr.Textbox(
                                    "This is a test of natural speech.", label="Text"
                                )
                                tts_voice = gr.Textbox(
                                    os.path.join(TTS_VOICES_PATH, "female1.wav"),
                                    label="Voice",
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

        if not args or args.sd:

            t2i_vid_button: gr.Button = None

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
                                step=32,
                                interactive=True,
                                label="Width",
                            )
                            t2i_height = gr.Slider(
                                minimum=256,
                                maximum=2048,
                                value=512,
                                step=32,
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
                                512, label="Width", precision=0, step=32
                            )
                            i2v_height = gr.Number(
                                512, label="Height", precision=0, step=32
                            )
                            i2v_fps = gr.Number(6, label="FPS", precision=0, minimum=1)
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
                            i2v_interpolation = gr.Number(
                                3, label="Frame Interpolation", precision=0, minimum=1
                            )

                        t2i_vid_button = gr.Button("Generate Video", interactive=False)

                        i2v_output = gr.Video(
                            None,
                            width=320,
                            height=320,
                            interactive=False,
                            label="Video",
                            format="mp4",
                            autoplay=True,
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
                                i2v_interpolation,
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
                            "robot assembly line", label="Audio description", lines=3
                        )
                        with gr.Row():
                            audiogen_duration = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=3,
                                step=1,
                                interactive=True,
                                label="Duration (seconds)",
                            )
                            audiogen_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.9,
                                value=1,
                                step=0.05,
                                interactive=True,
                                label="Temperature",
                            )
                        audiogen_button = gr.Button("Generate Audio")
                        audiogen_output = gr.Audio(interactive=False)
                        audiogen_button.click(
                            audiogen,
                            inputs=[
                                audiogen_prompt,
                                audiogen_duration,
                                audiogen_temperature,
                            ],
                            outputs=[audiogen_output],
                        )
                    with gr.Column():
                        musicgen_prompt = gr.TextArea(
                            "techno beat with a cool bassline",
                            label="Music description",
                            lines=3,
                        )
                        with gr.Row():
                            musicgen_duration = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=15,
                                step=1,
                                interactive=True,
                                label="Duration (seconds)",
                            )
                            musicgen_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.9,
                                value=1,
                                step=0.05,
                                interactive=True,
                                label="Temperature",
                            )
                        musicgen_button = gr.Button("Generate Music")
                        musicgen_output = gr.Audio(interactive=False)
                        musicgen_button.click(
                            musicgen,
                            inputs=[
                                musicgen_prompt,
                                musicgen_duration,
                                musicgen_temperature,
                            ],
                            outputs=[musicgen_output],
                        )
            with gr.Tab("Shap-e"):

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

                with gr.Row():
                    with gr.Column():
                        shap_e_prompt = gr.TextArea("a humanoid robot", label="Prompt")
                        shap_e_guidance = gr.Slider(
                            minimum=0,
                            maximum=50,
                            value=15,
                            step=0.1,
                            interactive=True,
                            label="Guidance",
                        )
                        shap_e_steps = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1,
                            interactive=True,
                            label="Steps",
                        )
                        shap_e_button = gr.Button("Generate")
                    with gr.Column():
                        shap_e_output = gr.Model3D(
                            None,
                            interactive=False,
                            label="Output",
                        )
                        shap_e_button.click(
                            shape_generate,
                            inputs=[
                                shap_e_prompt,
                                shap_e_steps,
                                shap_e_guidance,
                            ],
                            outputs=[shap_e_output],
                        )

        web_ui.queue().launch(
            prevent_thread_lock=prevent_thread_lock, inbrowser=args and not args.all
        )

        return web_ui


if __name__ == "__main__":
    print("Loading webui in main thread.")
    launch_webui(startup_args)
