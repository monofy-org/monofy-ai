import logging
import io
from llmclient.chat_utils import convert_gr_to_openai
from settings import LOG_LEVEL
from ttsclient import TTSClient
import gradio as gr
from llmclient import LLMClient

logging.basicConfig(level=LOG_LEVEL)

settings = {
    "language": "en",
    "speed": 1,
    "temperature": 0.75,
    "voice": "voices/female1.wav",
}


def launch_webui(use_llm=False, use_tts=False, prevent_thread_lock=False):
    tts: TTSClient = None
    llm: LLMClient = None    

    if use_tts:
        tts = TTSClient.instance

    if use_llm:
        llm = LLMClient.instance

    with gr.Blocks(analytics_enabled=False) as web_ui:
        with gr.Row():
            if use_tts:
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
                    grVoice = gr.Textbox("voices/female1.wav", label="Voice")
                    with gr.Row():
                        grSpeed = gr.Number("1", label="Speed")
                        grTemperature = gr.Number("0.75", label="Temperature")

                        grLanguage = gr.Dropdown(
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
                    grLanguage.change(set_language, inputs=[grLanguage])
                    grSpeed.change(set_speed, inputs=[grSpeed])
                    grTemperature.change(set_temperature, inputs=[grTemperature])
                    grVoice.change(set_voice, inputs=[grVoice])
                    grGenerateButton = gr.Button("Generate")
                    grAudioOutput = gr.Audio(
                        label="Audio Output",
                        type="numpy",
                        autoplay=True,                        
                        format="wav",
                        streaming=True
                    )

                    async def preview_speech(
                        text: str,
                        speed: int,
                        temperature: float,
                        voice: str,
                        language: str,
                    ):
                        # TODO stream to grAudio using generate_text_streaming
                        speech = tts.generate_speech(
                            text,
                            speed,
                            temperature,
                            voice,
                            language,
                        )
                        yield speech

                    grGenerateButton.click(
                        preview_speech,
                        inputs=[grText, grSpeed, grTemperature, grVoice, grLanguage],
                        outputs=[grAudioOutput]
                    )

            # Right half of the screen (Chat UI) - Only if use_llm is True
            if use_llm:
                grChatSpeak = None

                async def chat(
                    text: str,
                    history: list[list],                               
                    chunk_sentences=True
                ):
                    print(f"text={text}")
                    print(f"chunk_sentences={chunk_sentences}")

                    response = llm.chat(
                        text=text,
                        messages=convert_gr_to_openai(history),
                        chunk_sentences=chunk_sentences,
                    )

                    message = ""
                    for chunk in response:
                        if chunk_sentences:
                            message += " " + chunk

                            if tts and grChatSpeak.value:
                                print("")
                                logging.info("\nGenerating speech...")

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

    web_ui.launch(prevent_thread_lock=prevent_thread_lock)


if __name__ == "__main__":
    launch_webui(True)
