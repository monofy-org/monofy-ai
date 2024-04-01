import asyncio
from audioop import avg
import base64
import io
import struct
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.tts import TTSPlugin, TTSRequest
from plugins.voice_whisper import VoiceWhisperPlugin
from utils.audio_utils import resample


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
    plugins = [TTSPlugin, VoiceWhisperPlugin]    

    def __init__(self):
        super().__init__()        


@PluginBase.router.websocket("/voice/conversation")
async def voice_conversation(websocket: WebSocket):
    await websocket.accept()

    plugin: VoiceConversationPlugin = None

    try:
        plugin = await use_plugin(VoiceConversationPlugin)

        tts: TTSPlugin = None
        whisper: VoiceWhisperPlugin = None

        buffers: list[np.ndarray] = []

        while True:
            data = await websocket.receive_json()
            if not data:
                break
            action = data["action"]
            # print(data)
            if action == "call":
                await websocket.send_json({"status": "ringing"})
                tts = await use_plugin(TTSPlugin, True)
                await websocket.send_json({"status": "ringing"})
                tts.generate_speech_streaming("Hello.")
                await websocket.send_json({"status": "ringing"})
                whisper = await use_plugin(VoiceWhisperPlugin, True)
                await websocket.send_json({"status": "connected"})
            elif action == "end":
                await websocket.send_json({"status": "end"})
                break
            elif data["action"] == "audio":
                f32_array = np.frombuffer(base64.b64decode(data["data"]), dtype=np.float32)
                buffers.append(f32_array)
            elif data["action"] == "pause":
                audio = np.concatenate(buffers)                
                buffers = []
                sample_rate = data["sample_rate"]
                response = await whisper.process(audio, sample_rate)
                await websocket.send_json(response)
                for chunk in tts.generate_speech_streaming(TTSRequest(text=response["text"])):
                    await websocket.send_json({"audio": base64.b64encode(chunk.tobytes()).decode()})

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
