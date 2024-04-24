import os
import json
import logging
import time
import uuid
from typing import List, Optional
from pydantic import BaseModel
from fastapi import HTTPException, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
import yaml
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.text_utils import process_llm_text, remove_emojis
from utils.file_utils import fetch_pretrained_model
from settings import (
    LLM_GPU_SPLIT,
    LLM_MAX_NEW_TOKENS,
    LLM_DEFAULT_USER,
    LLM_DEFAULT_ASSISTANT,
    LLM_MAX_SEQ_LEN,
    LLM_MODEL,
    LLM_SCALE_ALPHA,
    LLM_SCALE_POS_EMB,
    LLM_STOP_CONDITIONS,
)


class ChatCompletionRequest(BaseModel):
    messages: list
    model: Optional[str] = "local"
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 20
    max_emojis: Optional[int] = 1  # -1 to disable, 0 = no emojis
    max_tokens: Optional[int] = LLM_MAX_NEW_TOKENS
    frequency_penalty: Optional[float] = 1.05
    presence_penalty: Optional[float] = 0.0
    stream: Optional[bool] = False
    context: Optional[str] = None
    bot_name: Optional[str] = LLM_DEFAULT_ASSISTANT
    user_name: Optional[str] = LLM_DEFAULT_USER


class ExllamaV2Plugin(PluginBase):

    name = "exllamav2"
    description = "ExLlamaV2 text generation for EXL2 models"
    instance = None
    plugins = ["TTSPlugin", "VoiceWhisperPlugin"]

    def __init__(self):

        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Config,
            ExLlamaV2Cache,
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import ExLlamaV2StreamingGenerator

        super().__init__()

        self.refresh_context()

        model_path = fetch_pretrained_model(LLM_MODEL)

        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        config.max_seq_len = LLM_MAX_SEQ_LEN
        config.scale_pos_emb = LLM_SCALE_POS_EMB
        config.scale_alpha_value = LLM_SCALE_ALPHA
        config.no_flash_attn = True

        # Still broken as of ExllamaV2 0.0.11, further research needed
        # LLM_GPU_SPLIT not supported with config.set_low_mem()
        # if LLM_GPU_SPLIT is None:
        #    config.set_low_mem()

        model = ExLlamaV2(config, lazy_load=True)
        cache = ExLlamaV2Cache(model, lazy=True)
        model.load_autosplit(cache, LLM_GPU_SPLIT)
        tokenizer = ExLlamaV2Tokenizer(config)
        # generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        streaming_generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

        self.resources["model"] = model
        self.resources["cache"] = cache
        self.resources["tokenizer"] = tokenizer
        self.resources["streaming_generator"] = streaming_generator

    def refresh_context(self):
        # read context.txt
        try:
            with open("context.txt", "r") as file:
                self.default_context = file.read()
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            self.default_context = f"Your name is {LLM_DEFAULT_ASSISTANT}. You are the default bot and you are super hyped about it. Considering the following conversation between User and Assistant, give a single response as Assistant. Do not prefix with your own name. Do not prefix with emojis."

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = 0.7,  # real default is 0.8
        top_k: int = 20,  # real default is 50
        top_p: float = 0.9,  # real default is 0.5
        token_repetition_penalty: float = 1.05,  # real default is 1.05
        typical: float = 1,
        stop_conditions: List[str] = None,
    ):
        try:
            from exllamav2 import ExLlamaV2Tokenizer
            from exllamav2.generator import (
                ExLlamaV2Sampler,
                ExLlamaV2StreamingGenerator,
            )

            tokenizer: ExLlamaV2Tokenizer = self.resources["tokenizer"]
            generator: ExLlamaV2StreamingGenerator = self.resources[
                "streaming_generator"
            ]
            assert tokenizer is not None
            assert generator is not None

            generator.set_stop_conditions(
                stop_conditions=[tokenizer.eos_token_id]
                + LLM_STOP_CONDITIONS
                + stop_conditions
            )

            settings = ExLlamaV2Sampler.Settings()
            settings.temperature = temperature
            settings.top_k = top_k
            settings.top_p = top_p
            settings.token_repetition_penalty = token_repetition_penalty
            settings.typical = typical

            settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

            time_begin = time.time()

            input_ids = tokenizer.encode(prompt)
            input_ids.to(self.device)

            full_token_count = len(input_ids[0])
            generated_tokens = 0
            message = ""

            generator.warmup()
            generator.begin_stream(input_ids, settings, True)

            while True:
                chunk, eos, _ = generator.stream()
                generated_tokens += 1
                chunk = process_llm_text(chunk, True)
                message = process_llm_text(message + chunk)

                yield chunk

                end_sentence = (
                    len(chunk) > 0
                    and chunk[-1] in ".?!\n"
                    and not chunk.endswith("Dr.")
                    and not chunk.endswith("Mr.")
                    and not chunk.endswith("Mrs.")
                    and not chunk.endswith("Ms.")
                    and not chunk.endswith("Capt.")
                    and not chunk.endswith("Cp.")
                    and not chunk.endswith("Lt.")
                    and not chunk.endswith("Mjr.")
                    and not chunk.endswith("Col.")
                    and not chunk.endswith("Gen.")
                    and not chunk.endswith("Prof.")
                    and not chunk.endswith("Sr.")
                    and not chunk.endswith("Jr.")
                    and not chunk.endswith("St.")
                    and not chunk.endswith("Ave.")
                    and not chunk.endswith("Blvd.")
                    and not chunk.endswith("Rd.")
                    and not chunk.endswith("Ct.")
                    and not chunk.endswith("Ln.")
                )

                if eos or (
                    generated_tokens + 16
                    >= max_new_tokens  # don't start a new sentence
                    and end_sentence
                    and message.count("```") % 2 == 0  # finish code block
                    and message.count("{") == message.count("}")  # finish JSON block
                ):
                    break

            del input_ids

            full_token_count += generated_tokens

            time_total = time.time() - time_begin

            logging.info(
                f"Generated {generated_tokens} tokens, {generated_tokens / time_total:.2f} tokens/second, {full_token_count} total tokens."
            )
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            raise e

    async def generate_chat_response(
        self,
        messages: List[dict],
        context: str = None,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
        token_repetition_penalty: float = 1.15,
        bot_name: str = None,
        user_name: str = LLM_DEFAULT_USER,
        stop_conditions: List[str] = [],
        max_emojis: int = 1,
    ):
        if not context:
            context = self.default_context

        if context and context.endswith(".yaml"):
            path = os.path.join("characters", context)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            # read from characters folder
            with open(path, "r") as file:
                yaml_data = yaml.safe_load(file.read())

            if not bot_name:
                bot_name = yaml_data.get("name", LLM_DEFAULT_ASSISTANT)

            context = yaml_data.get("context", context)

        if not bot_name:
            bot_name = LLM_DEFAULT_ASSISTANT

        context = (
            context.replace("{bot_name}", bot_name)
            .replace("{user_name}", user_name)
            .replace("{timestamp}", time.strftime("%A, %B %d, %Y %I:%M %p"))
        )

        prompt = f"system: {context}\n\n"

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            name = user_name if role == "user" else bot_name
            prompt += f"\n\n{name}: {content}"

        prompt += f"\n\n{bot_name}: "

        # combine response to string
        response = ""
        emoji_count = 0
        for chunk in self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            token_repetition_penalty=token_repetition_penalty,
            top_p=top_p,
            stop_conditions=[f"\n{bot_name}"] + stop_conditions,
        ):
            if max_emojis > -1:
                stripped_chunk = remove_emojis(chunk)
                if len(stripped_chunk) < len(chunk):
                    emoji_count += 1
                    if emoji_count > max_emojis:
                        chunk = stripped_chunk

            response += chunk

        return response.strip()


@PluginBase.router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    try:
        from exllamav2 import ExLlamaV2Tokenizer

        plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin)
        tokenizer: ExLlamaV2Tokenizer = plugin.resources["tokenizer"]

        content = ""
        token_count = 0

        response = await plugin.generate_chat_response(
            context=request.context,
            messages=request.messages,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            token_repetition_penalty=request.frequency_penalty,
            bot_name=request.bot_name,
            user_name=request.user_name,
        )

        emoji_count = 0

        for chunk in response:

            if request.max_emojis > -1:
                stripped_chunk = remove_emojis(chunk)
                if len(stripped_chunk) < len(chunk):
                    emoji_count += 1
                    if emoji_count > request.max_emojis:
                        chunk = stripped_chunk
            if len(chunk) > 0:
                content += chunk
                token_count += 1

        content = process_llm_text(content)

        response_tokens = tokenizer.encode(content).shape[0]

        response_data = dict(
            id=uuid.uuid4().hex,
            object="text_completion",
            created=int(time.time()),  # Replace with the appropriate timestamp
            model=request.model,
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                }
            ],
            usage={
                "prompt_tokens": 0,  # Replace with the actual prompt_tokens value
                "completion_tokens": response_tokens,
                "total_tokens": token_count,  # Replace with the actual total_tokens value
            },
        )

        release_plugin(ExllamaV2Plugin)
        return JSONResponse(content=response_data)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        release_plugin(ExllamaV2Plugin)
        raise HTTPException(status_code=500, detail=str(e))


@PluginBase.router.websocket("/chat/stream")
async def chat_streaming(websocket: WebSocket):
    await websocket.accept()

    data = await websocket.receive_text()
    req: ChatCompletionRequest = json.loads(data)

    plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin, True)

    for response in plugin.generate_text(req):
        yield response

    websocket.close()

    release_plugin(ExllamaV2Plugin)


@PluginBase.router.post("/chat/stream", response_class=StreamingResponse)
async def chat_streaming_post(req: ChatCompletionRequest):
    plugin = None
    try:
        plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin, True)
        return StreamingResponse(plugin.generate_text(req))
    finally:
        if plugin:
            release_plugin(ExllamaV2Plugin)
