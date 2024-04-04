import asyncio
import logging
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.exllamav2 import ExllamaV2Plugin
from plugins.tts import TTSPlugin, TTSRequest
from plugins.voice_whisper import VoiceWhisperPlugin
from utils.gpu_utils import clear_gpu_cache, set_seed
from utils.text_utils import process_text_for_tts


class VoiceHistoryItem(BaseModel):
    text: str
    speaker: str


class VoiceConversationRequest(BaseModel):
    context: str
    caller_name: str
    history: Optional[list[VoiceHistoryItem]] = None


class VoiceConversationPlugin(PluginBase):

    name = "Voice Conversation"
    description = "Voice conversation with a virtual assistant."
    instance = None
    plugins = ["TTSPlugin", "VoiceWhisperPlugin", "ExllamaV2Plugin"]

    def __init__(self):
        super().__init__()        

    async def warmup_speech(self, tts: TTSPlugin, voice="female1"):
        async for _ in tts.generate_speech_streaming(
            TTSRequest(text="Ok it's time to get started!", voice=voice)
        ):
            pass

    async def speak(self, websocket: WebSocket, tts: TTSPlugin, text: str, voice="female1"):

        text = process_text_for_tts(text)

        if not text:
            return

        async for chunk in tts.generate_speech_streaming(
            TTSRequest(text=text, voice=voice)
        ):
            await websocket.send_bytes(chunk.tobytes())

        


@PluginBase.router.websocket("/voice/conversation")
async def voice_conversation(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "ringing"})

    plugin: VoiceConversationPlugin = None

    set_seed(-1)

    try:
        plugin = await use_plugin(VoiceConversationPlugin)

        llm: ExllamaV2Plugin = None
        tts: TTSPlugin = None
        whisper: VoiceWhisperPlugin = None
        buffers: list[np.ndarray] = []
        next_action: str = None
        bot_name = "Alan"
        voice = "alan2"
        chat_history = []

        while True:

            if next_action == "audio":
                data = await websocket.receive_bytes()
                if not data:
                    break

                audio = np.frombuffer(data, dtype=np.float32)
                buffers.append(audio)
                next_action = None
                continue

            data = await websocket.receive_json()
            if not data:
                break

            action = data["action"]
            # print(data)
            if action == "call":
                if data.get("voice"):
                    voice = data["voice"]
                llm = await use_plugin(ExllamaV2Plugin, True)
                await websocket.send_json({"status": "ringing"})
                tts = await use_plugin(TTSPlugin, True)
                await websocket.send_json({"status": "ringing"})
                whisper = await use_plugin(VoiceWhisperPlugin, True)
                await plugin.warmup_speech(tts)
                await websocket.send_json({"status": "connected"})
                response = await llm.generate_chat_response(
                    chat_history,
                    bot_name=bot_name,
                    context="PhoneSupport.yaml",
                    max_new_tokens=100,
                    stop_conditions=["\n"],
                    max_emojis=0,
                )
                await websocket.send_json({"status": "ringing"})
                asyncio.create_task(plugin.speak(websocket, tts, response, voice))                
            elif action == "end":
                await websocket.send_json({"status": "end"})
                break
            elif data["action"] == "audio":
                next_action = "audio"
            elif data["action"] == "pause":
                audio = np.concatenate(buffers)
                buffers = []
                sample_rate = data["sample_rate"]
                transcription = await whisper.process(audio, sample_rate)

                if len(buffers) > 0:
                    print(
                        "Received additional buffers, consider skipping processsing..."
                    )

                text = transcription["text"].strip()
                if not text or len(text) < 3:
                    continue

                await websocket.send_json(transcription)

                chat_history.append({"role": "user", "content": text})
                response = await llm.generate_chat_response(
                    chat_history,
                    bot_name=bot_name,
                    context="PhoneSupport.yaml",
                    max_new_tokens=100,
                    stop_conditions=["\n"],
                    max_emojis=0,
                )
                chat_history.append({"role": "assistant", "content": response})
                hang_up = "[END CALL]" in response
                response = response.replace("[END CALL]", "")
                if hang_up:
                    await plugin.speak(websocket, tts, response, voice)
                else:
                    asyncio.create_task(plugin.speak(websocket, tts, response, voice))

                if hang_up:
                    logging.info("Other party ended the call.")
                    await websocket.send_json({"status": "end"})
                    break

            else:
                await websocket.send_json({"response": "Unknown action."})

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    finally:
        if plugin is not None:
            release_plugin(VoiceConversationPlugin)

        try:
            await websocket.close()
        except Exception:
            pass

        print("Call ended.")
        
        del buffers

        clear_gpu_cache()
