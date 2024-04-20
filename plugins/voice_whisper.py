from fastapi import WebSocket, WebSocketDisconnect
import numpy as np
from sklearn.pipeline import Pipeline
from modules.plugins import PluginBase, release_plugin, use_plugin
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

from utils.audio_utils import resample


class VoiceWhisperPlugin(PluginBase):

    name = "Voice Whisper"
    description = "Voice whispering with a virtual assistant."
    instance = None

    def __init__(self):

        super().__init__()

        model_name = "distil-whisper/distil-small.en"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="sdpa",
        )
        model.to(self.device)
        self.resources["model"] = model

        self.resources["processor"] = AutoProcessor.from_pretrained(model_name)
        self.resources["pipeline"] = pipeline(
            "automatic-speech-recognition",
            model=self.resources["model"],
            tokenizer=self.resources["processor"].tokenizer,
            feature_extractor=self.resources["processor"].feature_extractor,
            # max_new_tokens=128,
            # chunk_length_s=15,
            # batch_size=16,
            torch_dtype=self.dtype,
            device=self.device,
        )
        self.buffers = []

    async def process(self, audio: np.ndarray[np.float32], source_sample_rate: int):
        pipeline: Pipeline = self.resources["pipeline"]

        audio = resample(audio, source_sample_rate, 16000)

        response = pipeline(
            audio,
            chunk_length_s=30,
            batch_size=24,
            # generate_kwargs=generate_kwargs,
            return_timestamps=True,
        )

        print(f"Heard: {response["text"].strip()}")

        return response


@PluginBase.router.websocket("/voice/whisper")
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
                plugin.buffers.append(data["audio"])
            elif data["action"] == "speak_end":
                audio = np.concatenate(plugin.buffers)
                plugin.buffers = []
                sample_rate = data["sample_rate"]
                text = await plugin.process(audio, sample_rate)
                await websocket.send_json({"response": text})
            else:
                await websocket.send_json({"response": "Unknown action."})
    except WebSocketDisconnect:
        pass
    finally:
        if plugin is not None:
            await release_plugin(plugin)
