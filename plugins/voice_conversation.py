import asyncio
import base64
import io
import logging
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.tts import TTSPlugin, TTSRequest
from plugins.voice_whisper import VoiceWhisperPlugin
from utils.audio_utils import audio_to_base64


# disable logging for websockkets
logging.getLogger("websockets").setLevel(logging.ERROR)

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

        buffer = io.BytesIO()

        while True:
            data = await websocket.receive_json()
            if not data:
                break
            print(data)
            if data["action"] == "call":
                await websocket.send_json({"status": "ringing"})
                tts = await use_plugin(TTSPlugin, True)
                await websocket.send_json({"status": "ringing"})
                whisper = await use_plugin(VoiceWhisperPlugin, True)
                await websocket.send_json({"status": "connected"})
            elif data["action"] == "end":
                await websocket.send_json({"status": "end"})
                break
            elif data["action"] == "audio_chunk":
                
                base64_chunk = data["data"]
                buffer.write(base64.b64decode(base64_chunk))

                buffer_bytes = buffer.getvalue()

                if len(buffer_bytes) > 22050:                    
                    if max(buffer_bytes[-22050:]) < 0.2:
                        text = await whisper.process(buffer_bytes)
                        print("Text:", text)
                        buffer.close()
                        buffer = io.BytesIO()

                        for chunk in tts.generate_speech_streaming(
                            TTSRequest(text=text)
                        ):
                            print("Sending audio chunk...")
                            await websocket.send_json(
                                {"audio_chunk": audio_to_base64(chunk)}
                            )
            else:
                await websocket.send_json({"response": "Unknown action."})
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
