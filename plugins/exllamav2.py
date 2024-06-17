import json
import logging
import time
from typing import List, Optional
from pydantic import BaseModel
from fastapi import HTTPException, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.text_utils import (
    detect_end_of_sentence,
    format_chat_response,
    get_chat_context,
    process_llm_text,
    remove_emojis,
)
from utils.file_utils import cached_snapshot
from utils.gpu_utils import clear_gpu_cache
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

    def __init__(self, model_name_or_path: str = LLM_MODEL):

        super().__init__()

        self.current_model_name = None
        self.load_model(model_name_or_path)

    def load_model(self, model_name: str):

        if model_name == self.current_model_name:
            return

        from exllamav2 import (
            ExLlamaV2,
            ExLlamaV2Config,
            ExLlamaV2Cache,
            ExLlamaV2Tokenizer,
        )
        from exllamav2.generator import ExLlamaV2StreamingGenerator

        model: ExLlamaV2 = self.resources.get("model")

        if model is not None:
            logging.info(f"Unloading model {self.current_model_name}")
            model.free_device_tensors()
            del model
            self.resources["model"] = None
            clear_gpu_cache()

        self.current_model_name = model_name

        logging.info(f"Loading model {model_name}")

        if model_name.startswith("."):
            model_path = model_name
        else:
            model_path = cached_snapshot(model_name)

        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        config.no_flash_attn = True
        config.max_seq_len = LLM_MAX_SEQ_LEN
        if LLM_SCALE_POS_EMB:
            config.scale_pos_emb = LLM_SCALE_POS_EMB
        if LLM_SCALE_ALPHA:
            config.scale_alpha_value = LLM_SCALE_ALPHA

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
        streaming_generator.warmup()

        self.resources["model"] = model
        self.resources["cache"] = cache
        self.resources["tokenizer"] = tokenizer
        self.resources["streaming_generator"] = streaming_generator

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = 0.8,  # real default is 0.8
        top_k: int = 50,  # real default is 50
        top_p: float = 0.9,  # real default is 0.5
        token_repetition_penalty: float = 1.15,  # real default is 1.05
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

            settings.disallow_tokens(
                tokenizer,
                [tokenizer.eos_token_id, tokenizer.encode_special("<|eot_id|>")][0],
            )

            time_begin = time.time()

            input_ids = tokenizer.encode(prompt)
            input_ids.to(self.device)

            full_token_count = len(input_ids[0])
            generated_tokens = 0
            message = ""

            generator.begin_stream(input_ids, settings, True)

            while True:
                chunk, eos, _ = generator.stream()
                generated_tokens += 1
                chunk = process_llm_text(chunk, True)
                message = process_llm_text(message + chunk)

                yield chunk

                end_sentence = detect_end_of_sentence(chunk)

                if (
                    eos
                    or generated_tokens > max_new_tokens * 2
                    or (
                        generated_tokens + 16
                        >= max_new_tokens  # don't start a new sentence
                        and (
                            end_sentence
                            and message.count("```") % 2 == 0  # finish code block
                            and message.count("{")
                            == message.count("}")  # finish JSON block
                        )
                    )
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
        stream: bool = False,
    ):
        prompt = get_chat_context(messages, user_name, bot_name, context)

        # combine response to string
        response = ""
        emoji_count = 0

        if stream:
            for chunk in self.generate_text(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                token_repetition_penalty=token_repetition_penalty,
                stop_conditions=[f"\n{bot_name}"] + stop_conditions,
            ):
                if max_emojis > -1:
                    stripped_chunk = remove_emojis(chunk)
                    if len(stripped_chunk) < len(chunk):
                        emoji_count += 1
                        if emoji_count > max_emojis:
                            chunk = stripped_chunk

                yield chunk
        else:
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

            yield response.strip()


@PluginBase.router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    try:
        from exllamav2 import ExLlamaV2Tokenizer

        plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin)
        tokenizer: ExLlamaV2Tokenizer = plugin.resources["tokenizer"]

        content = ""
        emoji_count = 0

        prompt = get_chat_context(
            request.messages, request.user_name, request.bot_name, request.context
        )
        prompt_tokens = tokenizer.encode(prompt).shape[0]

        async for chunk in plugin.generate_chat_response(
            context=request.context,
            messages=request.messages,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            token_repetition_penalty=request.frequency_penalty,
            bot_name=request.bot_name,
            user_name=request.user_name,
        ):

            if request.max_emojis > -1:
                stripped_chunk = remove_emojis(chunk)
                if len(stripped_chunk) < len(chunk):
                    emoji_count += 1
                    if emoji_count > request.max_emojis:
                        chunk = stripped_chunk
            if len(chunk) > 0:
                content += chunk

        content = process_llm_text(content)

        completion_tokens = tokenizer.encode(content).shape[0]

        response_data = format_chat_response(
            content, request.model, prompt_tokens, completion_tokens
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
