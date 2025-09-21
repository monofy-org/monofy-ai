import base64
import logging
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from plugins.tts import TTSRequest, wav_to_mp3
from settings import LLM_MAX_SEQ_LEN, LLM_MODEL
from pathlib import Path
from plugins.tts import TTSPlugin
import json
import re
import time
import asyncio

from modules.plugins import PluginBase, use_plugin, release_plugin, use_plugin_unsafe
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer  # type: ignore
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler  # type: ignore

from modules.plugins import router
from utils.llm_utils import get_model

OpenAIPlugin = None  # Plugin reference used below


### === Schemas ===


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    functions: Optional[List[Dict[str, Any]]] = None    
    function_call: Optional[Any] = None
    stream: Optional[bool] = False
    tts: Optional[dict[str, str | float]] = None


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Delta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None    


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]


class StreamChoice(BaseModel):
    index: int
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChoice]


# === PLUGIN CLASS ===


class OpenAIPlugin(PluginBase):

    name = "Local OpenAI Plugin"
    plugins = ["TTSPlugin"]

    def __init__(self):
        super().__init__()        

    async def load_model(self, load_tts=False):
        if "model" not in self.resources:
            # **IMPORTANT**: Update this path to your model directory
            model_dir = get_model(LLM_MODEL)

            config = ExLlamaV2Config(model_dir)
            model = ExLlamaV2(config)
            cache = ExLlamaV2Cache(model, max_seq_len=LLM_MAX_SEQ_LEN, lazy=True)
            model.load_autosplit(cache, progress=True)
            tokenizer = ExLlamaV2Tokenizer(config)
            generator = ExLlamaV2DynamicGenerator(
                model=model,
                cache=cache,
                tokenizer=tokenizer,
                max_seq_len=LLM_MAX_SEQ_LEN,
            )

            self.resources["model"] = model
            self.resources["tokenizer"] = tokenizer
            self.resources["cache"] = cache
            self.resources["generator"] = generator

        if load_tts is True:
            self.resources["tts"] = use_plugin_unsafe(TTSPlugin)

    def get_weather(self, location: str):
        return {"location": location, "temperature": 22, "condition": "Sunny"}

    def format_prompt(
        self,
        messages: List[Message],
        sys_prompt = ""
    ):

        sys_prompt += (
            "\n### Here is the conversation so far:\nAssistant: How can I help?\n"
        )

        dialogue = ""
        for msg in messages:
            speaker = msg.name or msg.role
            dialogue += f"{speaker}: {msg.content}\n"

        print(sys_prompt + dialogue)
        return sys_prompt + dialogue + "\nAssistant:"

    def run_inference(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 512
    ):
        generator: ExLlamaV2DynamicGenerator = self.resources["generator"]

        # Note: ExLlamaV2's generate_simple is a straightforward way to generate text.
        # For more complex scenarios involving logits processing, you might need a different approach.
        output = generator.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop_conditions=[
                "[End]",
                "[END]",
                "\n###",
                "\nuser:",
                "\nUser:",
                "\nassistant:",
                "\nAssistant:",
                "\nUser (",
                "\nuser (",
                "\nsystem:",
                "\nSystem:",
            ],
        )

        return output[len(prompt) :].strip()

    async def stream_inference(self, prompt: str, temperature: float, max_tokens: int):
        generator: ExLlamaV2DynamicGenerator = self.resources["generator"]
        tokenizer: ExLlamaV2Tokenizer = self.resources["tokenizer"]

        gen_settings: ExLlamaV2Sampler.Settings = generator.get_default_settings()
        gen_settings.temperature = temperature

        job = ExLlamaV2DynamicJob(
            input_ids=tokenizer.encode(prompt, add_bos=True),
            max_new_tokens=max_tokens,
            stop_conditions=[tokenizer.eos_token_id],
            gen_settings=gen_settings,
            identifier=0,
        )
        generator.enqueue(job)

        # Reuse the same id for all chunks
        request_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        # Yield initial role chunk
        initial_chunk = ChatCompletionStreamResponse(
            id=request_id,
            object="chat.completion.chunk",
            created=created_time,
            model=LLM_MODEL,
            choices=[StreamChoice(index=0, delta=Delta(role="assistant"))],
        )
        yield f"event: message\ndata: {initial_chunk.model_dump_json(exclude_none=True)}\n\n"

        collected_text = ""
        while generator.num_remaining_jobs():
            results = generator.iterate()
            for result in results:
                text_chunk = result.get("text", "")
                collected_text += text_chunk

                if text_chunk:
                    response_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object="chat.completion.chunk",
                        created=created_time,
                        model=LLM_MODEL,
                        choices=[
                            StreamChoice(index=0, delta=Delta(content=text_chunk))
                        ],
                    )
                    yield f"event: message\ndata: {response_chunk.model_dump_json(exclude_none=True)}\n\n"

            await asyncio.sleep(0.01)

        # Final stop chunk
        final_chunk = ChatCompletionStreamResponse(
            id=request_id,
            object="chat.completion.chunk",
            created=created_time,
            model=LLM_MODEL,
            choices=[StreamChoice(index=0, delta=Delta(), finish_reason="stop")],
        )
        yield f"event: message\ndata: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

@router.post("/openai/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    plugin: OpenAIPlugin = None
    try:
        plugin = await use_plugin(OpenAIPlugin)
        await plugin.load_model(bool(request.tts))

        messages = request.messages
        prompt = plugin.format_prompt(messages)
        print(prompt, flush=True)

        if request.stream:
            return StreamingResponse(
                plugin.stream_inference(
                    prompt, request.temperature, request.max_tokens
                ),
                media_type="text/event-stream",
            )

        # Non-streaming inference
        response = plugin.run_inference(prompt, request.temperature, request.max_tokens)
        print(response, flush=True)

        answer = response

        choices = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ]

        if request.tts:
            tts: TTSPlugin = plugin.resources.get("tts")
            wav = tts.generate_speech(
                TTSRequest(
                    text=answer,
                    voice=request.tts.get("voice", "female1"),
                    language=request.tts.get("language", "en"),
                    pitch=request.tts.get("pitch", 1.0),
                    speed=request.tts.get("speed", 1.0),
                    temperature=request.tts.get("temperature", 0.75),
                )
            )
            mp3 = wav_to_mp3(wav)
            choices[0]["message"]["audio"] = base64.b64encode(mp3.getvalue()).decode("utf-8")

        return ChatCompletionResponse(
            id="chatcmpl-mistral-001",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=choices,
        )

    finally:
        if plugin:
            release_plugin(OpenAIPlugin)
