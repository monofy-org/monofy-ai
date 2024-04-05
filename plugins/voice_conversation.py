import asyncio
import logging
import os
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

SEARCH_CONTEXT = "Your job is to generate a JSON formatted response containing a ficticious but real-sounding person or business based on the request of a caller to a phone operator. The requests might seem silly or adult in nature but that's okay. Your response must be a JSON object or array of json objects starting with [{ name: ...\ncontaining only string values for the following properties: type (business or person), name, description, address, greeting (usually Hello for a person and the business name for a business). Make sure your response is valid JSON and that the response is for a person or business that is being requested. Begin your JSON output now. At the end, type [END]. Your output will be parsed, so do not include any extra information or additional fields that are not specifically mentioned here. Begin your JSON-only output now and don't forget to close objects, arrays, etc. and type [END] afterwards."


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

    async def speak(
        self, websocket: WebSocket, tts: TTSPlugin, text: str, voice="female1"
    ):

        text = process_text_for_tts(text)

        if not text:
            return

        async for chunk in tts.generate_speech_streaming(
            TTSRequest(text=text, voice=voice)
        ):
            try:
                await websocket.send_bytes(chunk.tobytes())
            except WebSocketDisconnect:
                break


@PluginBase.router.websocket("/voice/conversation")
async def voice_conversation(websocket: WebSocket):
    set_seed(-1)

    plugin: VoiceConversationPlugin = None

    try:
        plugin = await use_plugin(VoiceConversationPlugin)
        task = asyncio.create_task(conversation_loop(plugin, websocket))
        await task
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(e, exc_info=True)

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

        clear_gpu_cache()


async def conversation_loop(plugin: VoiceConversationPlugin, websocket: WebSocket):

    llm: ExllamaV2Plugin = None
    tts: TTSPlugin = None
    whisper: VoiceWhisperPlugin = None
    buffers: list[np.ndarray] = []
    next_action: str = None
    bot_name = None  # use default
    voice = "female1"
    chat_history = []
    context = "phone/Default.yaml"

    await websocket.accept()
    await websocket.send_json({"status": "ringing"})

    while True:

        if next_action == "audio":
            data = await websocket.receive_bytes()
            if not data:
                break

            audio = np.frombuffer(data, dtype=np.float32)
            buffers.append(audio)
            next_action = None
            continue

        try:
            data = await websocket.receive_json()
        except Exception as e:
            logging.warn(e)
            break

        if not data:
            break

        action = data["action"]
        # print(data)
        if action == "call":
            if data.get("number") == "0":
                context = "phone/Operator.yaml"

            greeting = None

            # read yaml file as string
            with open(os.path.join("characters", context), "r") as f:
                y = f.read()
                sp = y.split("greeting: ")
                if len(sp) > 1:
                    greeting = sp[1].split("\n")[0]
                    chat_history.append({"role": "assistant", "content": greeting})

            if data.get("voice"):
                voice = data["voice"]
            llm = await use_plugin(ExllamaV2Plugin, True)
            await websocket.send_json({"status": "ringing"})
            tts = await use_plugin(TTSPlugin, True)
            await websocket.send_json({"status": "ringing"})
            whisper = await use_plugin(VoiceWhisperPlugin, True)
            await plugin.warmup_speech(tts)
            await websocket.send_json({"status": "ringing"})
            response = (
                greeting
                if greeting
                else await llm.generate_chat_response(
                    chat_history,
                    bot_name=bot_name,
                    context=context,
                    max_new_tokens=100,
                    stop_conditions=["\n"],
                    max_emojis=0,
                )
            )            
            await websocket.send_json({"status": "connected"})
            asyncio.create_task(plugin.speak(websocket, tts, response, voice))
        elif action == "end":
            await websocket.send_json({"status": "end"})
            break
        elif data["action"] == "audio":
            next_action = "audio"
        elif data["action"] == "speech":
            tts.interrupt = True
            buffers = []
        elif data["action"] == "pause":
            audio = np.concatenate(buffers)
            buffers = []
            sample_rate = data["sample_rate"]
            transcription = await whisper.process(audio, sample_rate)

            if len(buffers) > 0:
                print("Received additional buffers, consider skipping processsing...")

            text = transcription["text"].strip()
            if not text or len(text) < 3:
                continue

            await websocket.send_json(transcription)

            chat_history.append({"role": "user", "content": text})

            response = await llm.generate_chat_response(
                chat_history,
                bot_name=bot_name,
                context=context,
                max_new_tokens=100,
                stop_conditions=["\n"],
                max_emojis=0,
            )
            chat_history.append({"role": "assistant", "content": response})
            hang_up = "[END CALL]" in response
            transfer = "[TRANSFER]" in response
            search = "[SEARCH]" in response
            response = (
                response.replace("[END CALL]", "")
                .replace("[TRANSFER]", "")
                .replace("[SEARCH]", "")
            )
            if hang_up:
                await plugin.speak(websocket, tts, response, voice)
            else:
                asyncio.create_task(plugin.speak(websocket, tts, response, voice))

            if hang_up:
                logging.info("Other party ended the call.")
                await websocket.send_json({"status": "end"})
                break
            elif search:
                search_response = await llm.generate_chat_response(
                    chat_history,
                    SEARCH_CONTEXT,
                    max_new_tokens=200,
                    max_emojis=0,
                    temperature=0.6,
                )
                chat_history.append({"role": "system", "content": "Help the customer choose from the following:\n\n" + search_response})
                print(search_response)
                break
            elif transfer:
                # TODO: Implement transfer
                break
        else:
            await websocket.send_json({"response": "Unknown action."})
            break

        await asyncio.sleep(0.01)
