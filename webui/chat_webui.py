import base64
import io
import os
import gradio as gr
from modules.plugins import use_plugin
from modules.webui import webui
from plugins.openai import ChatCompletionRequest, chat_completion
from plugins.tts import get_voices, get_languages
from utils.llm_utils import get_character, get_characters

# chat_history = [{ "role": "system", "content": "You are a helpful assistant named Stacy." }]
audio_buffer = None

tts_languages = get_languages()
character_list = get_characters()


@webui()
def add_interface(*args, **kwargs):
    async def func(
        text: str,
        history: list[list],
        bot_name: str,
        context: str,
        use_tts: bool,
        voice: str,
        language: str,
        pitch: float,
        speed: float,
        temperature: float,
    ):

        global audio_buffer

        history.append({"role": "user", "content": text})

        content = context.replace("{bot_name}", bot_name).replace("{name}", bot_name)

        sys_msg = [{"role": "system", "content": content}]

        response = await chat_completion(
            ChatCompletionRequest(
                model="default",
                messages=sys_msg + history,
                tts=(
                    {
                        "voice": voice,
                        "language": tts_languages.get(language, "en"),
                        "pitch": pitch,
                        "speed": speed,
                        "temperature": temperature,
                    }
                    if use_tts
                    else None
                ),
            )
        )
        message: dict = response.choices[0]["message"]
        audio = message.get("audio")

        if audio:
            audio_buffer = base64.b64decode(audio)
            # save to "./cache/chatbot_buffer.mp3"
            with open("./.cache/chatbot_buffer.mp3", "wb") as f:
                f.write(audio_buffer)

        print(message["role"], message["content"])
        yield gr.ChatMessage(content=message["content"], role=message["role"]), (
            gr.Audio("./.cache/chatbot_buffer.mp3", visible=True)
            if audio
            else gr.Audio(visible=False)
        )

    tab = gr.Tab("Chat")

    with tab:

        with gr.Row():

            with gr.Column():

                filename = (
                    "Default.yaml"
                    if "Default.yaml" in character_list
                    else character_list[0] if character_list else None
                )
                if not filename:
                    gr.Error(
                        "No characters found. Please add a character YAML file in the 'characters' folder."
                    )
                    filename = "Default.yaml"

                with gr.Row():
                    with gr.Column(scale=2):
                        character = gr.Dropdown(
                            choices=get_characters(),
                            value=filename,
                            allow_custom_value=True,
                            label="Character",
                        )
                    with gr.Column():
                        refresh = gr.Button("Refresh", size="sm")
                        save = gr.Button("Save", size="sm")

                audio = gr.Audio(
                    label="Chatbot Response",
                    autoplay=True,
                    format="mp3",
                    # visible=False,
                )

                with gr.Accordion("Speech", open=False):
                    tts_checkbox = gr.Checkbox(
                        label="Enable speech",
                        value=False,
                    )
                    tts_pitch = gr.Slider(
                        value=0, label="Pitch", minimum=-1200, maximum=1200, precision=0
                    )
                    tts_speed = gr.Slider(
                        value=1.0,
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                    )
                    tts_temperature = gr.Slider(
                        value=0.75,
                        label="Temperature",
                        minimum=0.5,
                        maximum=2.0,
                    )

                with gr.Accordion("Character"):
                    name = gr.Textbox(value="Assistant", label="Bot Name")
                    with gr.Row():
                        tts_voice = gr.Dropdown(
                            get_voices(), value="female1", label="Voice"
                        )
                        tts_lang = gr.Dropdown(
                            choices=tts_languages.keys(),
                            value="English",
                            label="Language (Accent)",
                        )
                    context = gr.TextArea(
                        "You are a helpful assistant.", lines=10, label="Context"
                    )

            with gr.Column(scale=2):

                chat = gr.ChatInterface(
                    fn=func,
                    type="messages",
                    additional_inputs=[
                        name,
                        context,
                        tts_checkbox,
                        tts_voice,
                        tts_lang,
                        tts_pitch,
                        tts_speed,
                        tts_temperature,
                    ],
                    additional_outputs=[audio],
                )

    def refresh_characters():
        global character_list
        character_list = get_characters()
        return gr.Dropdown(choices=character_list)

    def update_context(yaml_file: str):
        if not yaml_file:
            return gr.Error("No character selected")
        content = get_character(yaml_file)
        return (
            gr.Textbox(content.get("name", "Assistant")),
            gr.TextArea(content.get("context", "You are a helpful assistant.")),
            gr.Dropdown(value=content.get("tts_voice", "female1")),
            gr.Dropdown(value=content.get("tts_language", "English")),
        )

    refresh.click(
        refresh_characters,
        outputs=[character],
    )

    character.change(
        update_context,
        inputs=[character],
        outputs=[name, context, tts_voice, tts_lang],
    )


add_interface()
