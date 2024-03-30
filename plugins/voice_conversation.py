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

    def __init__(self):

        super().__init__()


@PluginBase.router.websocket("/voice/conversation", tags=["Voice"])
async def voice_conversation(websocket: WebSocket):
    await websocket.accept()

    plugin: VoiceConversationPlugin = None

    try:
        plugin = await use_plugin(VoiceConversationPlugin)

        tts: TTSPlugin = await use_plugin(TTSPlugin, True)
        whisper: VoiceWhisperPlugin = await use_plugin(VoiceWhisperPlugin, True)

        while True:
            data = await websocket.receive_json()
            print(data)
            if data["action"] == "call":                
                await websocket.send_json({"status": "ringing"})
            elif data["action"] == "end":
                await websocket.send_json({"status": "end"})
                break
            elif data["action"] == "audio_chunk":
                transcription = whisper.process(data["data"])
                print("Transcription:", transcription)
                for chunk in tts.generate_speech_streaming(
                    TTSRequest(text=transcription)
                ):
                    await websocket.send_json({"audio_chunk": audio_to_base64(chunk)})
            else:
                await websocket.send_json({"response": "Unknown action."})
    except WebSocketDisconnect:
        pass
    finally:
        if plugin is not None:
            await release_plugin(VoiceConversationPlugin)

        await websocket.close()
        print("Call ended.")
