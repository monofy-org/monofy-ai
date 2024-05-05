import logging
from typing import Literal
import gradio as gr
from modules.webui import webui
from modules.plugins import release_plugin, use_plugin
from plugins.tts import TTSPlugin, TTSRequest
from plugins.extras.tts_edge import generate_speech_edge, tts_edge_voices


@webui(section="TTS")
def add_interface(*args, **kwargs):
    async def func(
        model: Literal["edge-tts", "xtts"],
        text: str,
        voice: str,
        speed: float,
    ):
        req: TTSRequest = TTSRequest(
            text=text,
            speed=speed,
            format="mp3",
            stream=True,
        )
        if model == "edge-tts":
            async for chunk in generate_speech_edge(text, voice, speed):
                yield chunk

        else:
            plugin: TTSPlugin = None

            try:
                plugin: TTSPlugin = await use_plugin(TTSPlugin)

                async for chunk in plugin.generate_speech_streaming(req):
                    yield chunk

            except Exception as e:
                logging.exception(e)
                raise e

            finally:
                if plugin is not None:
                    release_plugin(plugin)

    tab = gr.Tab(
        label="Text-to-Speech",
    )

    with tab:
        with gr.Row():
            tts_model = gr.Radio(
                label="Model",
                choices=["edge-tts", "xtts"],
                value="edge-tts",
            )
            tts_input = gr.Textbox(
                label="Text",
                value="Hello, world!",
            )
            tts_voice = gr.Dropdown(
                label="Voice",
            )
            tts_speed = gr.Slider(
                label="Speed",
                value=1.0,
                minimum=0.5,
                maximum=2.0,
                step=0.1,
            )
            tts_button = gr.Button("Generate Speech")
            tts_audio = gr.Audio(
                label="Audio",
                type="numpy",
                format="mp3",
                interactive=False,
                streaming=True,
                autoplay=True,
            )
            tts_button.click(
                fn=func,
                inputs=[tts_model, tts_input, tts_voice, tts_speed],
                outputs=[tts_audio],
            )

    async def update_voices():
        voices = await tts_edge_voices()
        return gr.Dropdown(
            choices=[x["ShortName"] for x in voices], value="en-US-AvaNeural"
        )

    tts_voice.attach_load_event(update_voices, None)


add_interface()
