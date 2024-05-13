import gradio as gr
from modules.plugins import use_plugin
from modules.webui import webui
from plugins.exllamav2 import ExllamaV2Plugin
from utils.chat_utils import convert_gr_to_openai

chat_history = []


@webui()
def add_interface(*args, **kwargs):
    async def func(text: str, history: list[list], tts: bool):

        plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin)

        message = ""

        entry = [text, ""]

        history.append(entry)

        async for chunk in plugin.generate_chat_response(
            messages=convert_gr_to_openai(history),
            stream=True,
        ):
            message += chunk
            entry[1] = message
            yield message

        

    tab = gr.Tab("Chat")

    with tab:

        tts_checkbox = gr.Checkbox(
            label="Text-to-speech",
            value=False,            
        )

        chat = gr.ChatInterface(
            fn=func,
            additional_inputs=[
                tts_checkbox,
            ],
            fill_height=True
        )


add_interface()
