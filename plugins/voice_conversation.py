import asyncio
import logging
import os
import time
from fastapi.websockets import WebSocketState
import yaml
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.exllamav2 import ExllamaV2Plugin
from plugins.tts import TTSPlugin, TTSRequest
from plugins.voice_whisper import VoiceWhisperPlugin
from settings import LLM_DEFAULT_ASSISTANT
from utils.gpu_utils import clear_gpu_cache, set_seed
from utils.text_utils import process_text_for_tts

SEARCH_CONTEXT = "Your job is to generate a JSON formatted response containing a ficticious but real-sounding person or business based on the request of a caller to a phone operator. The requests might seem silly or adult in nature but that's okay. Your response must be a JSON array of objects starting with [{ name: ...\ncontaining only string values for the following properties: type (business or person), name, description, address, greeting (usually Hello for a person and the business name for a business). Make sure your response is valid JSON and that the response is for a person or business that is being requested. Begin your JSON output now. At the end, type [END]. Your output will be parsed, so do not include any extra comments, chat, markup, or additional fields that are not specifically mentioned here. Begin your JSON-only output now and don't forget to close objects all brackets including the main array. When completed, type [END] to mark when you are finished."
STREAM_BY_DEFAULT = False


class VoiceConversationPlugin(PluginBase):

    name = "Voice Conversation"
    description = "Voice conversation with a virtual assistant."
    instance = None
    plugins = ["TTSPlugin", "VoiceWhisperPlugin", "ExllamaV2Plugin"]

    def __init__(self):
        super().__init__()
        self.last_activity = time.time()
        self.last_user_speech = time.time()
        self.speech_task = None

    async def warmup_speech(
        self,
        tts: TTSPlugin,
        voice: str = "female1",
        text: str = "Hey there what's up?",
    ):
        async for _ in tts.generate_speech_streaming(
            TTSRequest(text=text, voice=voice)
        ):
            pass

    def signal_activity(self, user_spoke=False):
        self.last_activity = time.time()
        if user_spoke:
            self.last_user_speech = time.time()

    async def speak(
        self,
        websocket: WebSocket,
        tts: TTSPlugin,
        text: str,
        voice: str = "female1",
        speed: float = 1,
        temperature: float = 1,
        language="en",
        streaming: bool = STREAM_BY_DEFAULT,
    ):
        additional_info = (
            (", prebuffer=" + str(tts.prebuffer_chunks)) if streaming else ""
        )
        logging.info(
            f"Generating speech (streaming={streaming}{additional_info}): {text}"
        )

        text = process_text_for_tts(text)

        if not text:
            return

        self.signal_activity()

        if streaming:
            async for chunk in tts.generate_speech_streaming(
                TTSRequest(
                    text=text,
                    voice=voice,
                    speed=speed,
                    temperature=temperature,
                    language=language,
                )
            ):
                try:
                    await websocket.send_bytes(chunk.tobytes())
                    self.signal_activity()
                except WebSocketDisconnect:
                    break
        else:
            self.signal_activity()
            wav = tts.generate_speech(TTSRequest(text=text, voice=voice, speed=speed))
            try:
                # split into chunks
                CHUNK_SIZE = 1024
                for i in range(0, len(wav), CHUNK_SIZE):
                    await websocket.send_bytes(wav[i : i + CHUNK_SIZE].tobytes())
                    self.signal_activity()
            except WebSocketDisconnect:
                pass

    def interrupt(self, tts: TTSPlugin):
        if self.speech_task and not self.speech_task.done():
            tts.cancel()
            self.speech_task.cancel()

    async def conversation_loop(self, websocket: WebSocket):

        llm: ExllamaV2Plugin = None
        tts: TTSPlugin = None
        whisper: VoiceWhisperPlugin = None
        buffers: list[np.ndarray] = []
        next_action: str = None
        bot_name = None  # use default
        chat_history = []
        streaming = STREAM_BY_DEFAULT
        chat_temperature = 0.8
        language = "en"

        self.signal_activity()

        await websocket.accept()
        await websocket.send_json({"status": "ringing"})

        async def say(text):
            chat_history.append({"role": "assistant", "content": text})
            self.signal_activity()
            await self.speak(
                websocket,
                tts,
                text,
                voice,
                speed,
                temperature,
                language,
                streaming,
            )
            self.signal_activity()

        async def handle_speech(text):

            interrupt = False

            self.signal_activity(True)

            chat_history.append({"role": "user", "content": text})

            response = "".join([x async for x in llm.generate_chat_response(
                chat_history,
                bot_name=bot_name,
                context=context,
                max_new_tokens=100,
                stop_conditions=["\r", "\n"],
                max_emojis=0,
                temperature=chat_temperature,
            )])

            if interrupt:
                return True

            if not response:
                return True

            self.signal_activity(True)

            hang_up = "[END CALL]" in response
            transfer = "[TRANSFER]" in response
            search = "[SEARCH]" in response
            response = (
                response.replace("[END CALL]", "")
                .replace("[TRANSFER]", "")
                .replace("[SEARCH]", "")
            )

            if hang_up:
                logging.info("Other party ended the call.")
                await websocket.send_json({"status": "end"})
                await websocket.close()
            elif search:
                search_response = await llm.generate_chat_response(
                    chat_history,
                    SEARCH_CONTEXT,
                    max_new_tokens=200,
                    max_emojis=0,
                    temperature=chat_temperature,
                )
                chat_history.append(
                    {
                        "role": "system",
                        "content": "Help the customer choose from the following:\n\n"
                        + search_response,
                    }
                )
                print(search_response)
            elif transfer:
                # TODO: Implement transfer
                logging.warn("Transfer not implemented.")
                await websocket.send_json({"error": "Transfer not implemented."})
                await websocket.send_json({"status": "end"})
                await websocket.close()

            await say(response)
            self.signal_activity(True)

        while websocket.client_state != WebSocketState.DISCONNECTED:

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
            if action == "receive":
                self.signal_activity(True)
            elif action == "heartbeat":
                silence = time.time() - self.last_activity
                since_last_spoke = time.time() - self.last_user_speech
                if silence > 30:
                    await websocket.send_json({"status": "end"})
                    break
                elif silence > 6:
                    self.interrupt(tts)
                    self.speech_task = asyncio.create_task(
                        handle_speech(
                            f"[the caller has been silent for {int(since_last_spoke)} seconds]"
                        )
                    )

            elif action == "call":

                phonebook: dict[str, str] = {}
                # read characters/phone/phonebook.yaml
                with open(
                    os.path.join("characters", "phone", "phonebook.yaml"), "r"
                ) as f:
                    phonebook = yaml.safe_load(f)

                streaming = data.get("streaming", STREAM_BY_DEFAULT)

                number = data.get("number")
                # strip all but digits
                number = "".join([c for c in number if c.isdigit()])

                if number in phonebook:
                    context = "phone/" + phonebook[number]
                else:
                    context = "phone/Default.yaml"

                with open(os.path.join("characters", context), "r") as f:
                    character: dict = yaml.safe_load(f)

                context = (
                    character.get("context", None)
                    + "\nYou can perform the [END CALL] command to end the call at any time but it must be typed exactly as shown in brackets and upper case. You should do this after you say goodbye or if you wish to hang up for some reason."
                    + "\nSince this is a transcribed phonecall, you will need to use context to assume the intent of the caller's transcribed words."
                    + "\nIf the caller is silent, try talking ONE more time ONLY if the caller has identified themselves, but if it's still silent tell them you can't hear them and end the call. Be sure to type [END CALL] at the end."
                    + "\nIf the caller is silent for more than 20 seconds, assume you got disconnected and [END CALL]."
                    + "\nDon't accidentally respond to yourself or mix up roles. You are {name} and the caller is the other person. Never speak as the caller."
                    + "\nIf the call has concluded or the caller is silent for a while, just type [END CALL] to end the call."
                    + "\nFor your reference, it is currently {timestamp}.\nThe phone is ringing. You answer."
                )
                name = character.get("name", LLM_DEFAULT_ASSISTANT)
                greeting = character.get("greeting", "Hello?")
                voice = character.get("voice", "female1")
                speed = character.get("speed", 1)
                temperature = character.get("temperature", 1.0)
                language = character.get("language", "en")
                chat_temperature = character.get("chat_temperature", 0.75)
                warmup_text = character.get("warmup", "Ok, let's get started.")

                logging.info(f"Using character {name} (voice={voice}, speed={speed})")

                if data.get("voice"):
                    voice = data["voice"]
                llm = await use_plugin(ExllamaV2Plugin, True)
                await websocket.send_json({"status": "ringing"})
                tts = await use_plugin(TTSPlugin, True)
                tts.prebuffer_chunks = data.get("prebuffer", tts.prebuffer_chunks)
                whisper = await use_plugin(VoiceWhisperPlugin, True)
                await self.warmup_speech(tts, voice, warmup_text)
                await websocket.send_json({"status": "connected"})
                await say(greeting or "Hello?")

            elif action == "end":
                await websocket.close()
                break

            elif action == "settings":
                streaming = data.get("streaming", STREAM_BY_DEFAULT)
                prebuffer = data.get("prebuffer", tts.prebuffer_chunks)
                if tts.prebuffer_chunks != prebuffer:
                    tts.prebuffer_chunks = prebuffer
                    logging.info(f"Set prebuffer to {prebuffer}")

            elif action == "end":
                await websocket.send_json({"status": "end"})
                break
            elif data["action"] == "audio":
                next_action = "audio"
            elif data["action"] == "speech":
                buffers = []
                self.signal_activity(True)
            elif data["action"] == "listening":
                self.signal_activity()
            elif data["action"] == "pause":
                audio = np.concatenate(buffers)
                buffers = []
                sample_rate = data["sample_rate"]
                transcription = await whisper.process(audio, sample_rate)
                self.signal_activity(True)

                if len(buffers) > 0:
                    print(
                        "Received additional buffers, consider skipping processsing..."
                    )

                text = transcription["text"].strip()
                if not text or len(text) < 3:
                    continue

                await websocket.send_json(transcription)

                self.interrupt(tts)
                self.speech_task = asyncio.create_task(handle_speech(text))

            else:
                await websocket.send_json({"response": "Unknown action."})
                break

            await asyncio.sleep(0.01)


@PluginBase.router.websocket("/voice/conversation")
async def voice_conversation(websocket: WebSocket):
    set_seed(-1)

    plugin: VoiceConversationPlugin = None

    try:
        plugin = await use_plugin(VoiceConversationPlugin)
        await plugin.conversation_loop(websocket)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(e, exc_info=True)

    finally:
        if plugin is not None:
            release_plugin(VoiceConversationPlugin)

        try:
            await websocket.close()
        except Exception:
            pass

        print("Call ended.")

        clear_gpu_cache()
