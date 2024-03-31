from audioop import avg
import base64
import io
import struct
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.tts import TTSPlugin, TTSRequest
from plugins.voice_whisper import VoiceWhisperPlugin
from utils.audio_utils import audio_to_base64


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

        buffer = io.BytesIO()        

        while True:
            data = await websocket.receive_json()
            if not data:
                break
            # print(data)
            if data["action"] == "call":
                await websocket.send_json({"status": "ringing"})
                tts = await use_plugin(TTSPlugin, True)
                await websocket.send_json({"status": "ringing"})
                whisper = await use_plugin(VoiceWhisperPlugin, True)
                await websocket.send_json({"status": "connected"})
            elif data["action"] == "end":
                await websocket.send_json({"status": "end"})
                break
            elif data["action"] == "audio":

                f32_arr = data["data"]
                sample_rate = data["sample_rate"] or 16000
                bytes_arr = struct.pack("%sf" % len(f32_arr), *f32_arr)
                buffer.write(bytes_arr)

                buffer_bytes = buffer.getvalue()
                buffer_len = len(buffer_bytes)                

                if buffer_len > sample_rate:
                    max_amplitude = avg(buffer_bytes, 4)
                    print("Buffer length:", buffer_len)
                    if max_amplitude < 0.5:
                        print("Silence detected.")
                        text = await whisper.process(buffer_bytes)
                        print("Text:", text)
                        buffer.close()
                        buffer = io.BytesIO()

                        for chunk in tts.generate_speech_streaming(
                            TTSRequest(text=text)
                        ):
                            if not chunk:
                                break

                            print("Sending audio chunk...")
                            await websocket.send_json(
                                {"action": "audio", "data": chunk}
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
