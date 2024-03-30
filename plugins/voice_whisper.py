from fastapi import WebSocket, WebSocketDisconnect
from modules.plugins import PluginBase, release_plugin, use_plugin
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class VoiceWhisperPlugin(PluginBase):

    name = "Voice Whisper"
    description = "Voice whispering with a virtual assistant."
    instance = None

    def __init__(self):

        super().__init__()

        self.resources["processor"] = WhisperProcessor.from_pretrained(
            "openai/whisper-tiny"
        )

        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        model.config.forced_decoder_ids = None

    def process(self, audio):
        model: WhisperForConditionalGeneration = self.resources["model"]
        processor: WhisperProcessor = self.resources["processor"]
        input_features = processor(
            audio, sampling_rate=22050, return_tensors="pt"
        ).input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
        return transcription


@PluginBase.router.websocket("/voice/whisper", tags=["Voice"])
async def voice_whisper(websocket: WebSocket):
    await websocket.accept()

    plugin: VoiceWhisperPlugin = None

    try:
        plugin = await use_plugin(VoiceWhisperPlugin)

        while True:
            data = await websocket.receive_json()
            print(data)
            if data["action"] == "start":
                await websocket.send_json({"response": "Starting whispering..."})
            elif data["action"] == "stop":
                await websocket.send_json({"response": "Stopping whispering..."})
                break
            elif data["action"] == "audio":
                transcription = plugin.process(data["audio"])
                await websocket.send_json({"transcription": transcription})
            else:
                await websocket.send_json({"response": "Unknown action."})
    except WebSocketDisconnect:
        pass
    finally:
        if plugin is not None:
            await release_plugin(plugin)
