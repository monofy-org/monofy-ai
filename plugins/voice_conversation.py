import asyncio
import random
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.exllamav2 import ExllamaV2Plugin
from plugins.tts import TTSPlugin, TTSRequest
from plugins.voice_whisper import VoiceWhisperPlugin
from submodules.TTS.TTS.demos.xtts_ft_demo.xtts_demo import clear_gpu_cache


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
    plugins = ["TTSPlugin", "VoiceWhisperPlugin"]

    def __init__(self):
        super().__init__()

    async def speak(self, websocket: WebSocket, tts: TTSPlugin, text: str):
        async for chunk in tts.generate_speech_streaming(TTSRequest(text=text)):
            await websocket.send_bytes(chunk.tobytes())


@PluginBase.router.websocket("/voice/conversation")
async def voice_conversation(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"status": "ringing"})

    plugin: VoiceConversationPlugin = None

    try:
        plugin = await use_plugin(VoiceConversationPlugin)

        llm: ExllamaV2Plugin = None
        tts: TTSPlugin = None
        whisper: VoiceWhisperPlugin = None

        buffers: list[np.ndarray] = []

        chat_history = []

        next_action: str = None

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
                llm = await use_plugin(ExllamaV2Plugin, True)
                await websocket.send_json({"status": "ringing"})
                tts = await use_plugin(TTSPlugin, True)
                await websocket.send_json({"status": "ringing"})
                whisper = await use_plugin(VoiceWhisperPlugin, True)
                await websocket.send_json({"status": "connected"})
                message = random.choice(["Hello?", "Hello!", "Hey!", "Hey there!"])
                chat_history.append({"role": "assistant", "content": message})
                await websocket.send_json({"status": "ringing"})
                await plugin.speak(websocket, tts, message)
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
                    bot_name="Stacy",
                    context="PhoneSupport.yaml",
                    max_new_tokens=80,
                )
                await plugin.speak(websocket, tts, response)

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

        del chat_history
        del buffers

        clear_gpu_cache()
