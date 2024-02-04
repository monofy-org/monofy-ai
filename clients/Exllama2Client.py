import gc
from typing import AsyncGenerator
import logging

import torch
from utils.file_utils import fetch_pretrained_model
from utils.gpu_utils import load_gpu_task, autodetect_device
from utils.text_utils import process_llm_text
from settings import (
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL,
    LLM_GPU_SPLIT,
    LLM_MAX_SEQ_LEN,
    LLM_SCALE_ALPHA,
    LLM_SCALE_POS_EMB,
    LLM_STOP_CONDITIONS,
)
import time
from typing import List
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler,
)
from clients import Exllama2Client


friendly_name = "exllamav2"
logging.warn(f"Initializing {friendly_name}...")

current_model_name = LLM_MODEL
model_path = None
model = None
config = None
cache = None
tokenizer = None
generator = None
streaming_generator = None
user_name = "User"
assistant_name = "Assistant"


def read_context_file(from_api: bool = False):
    try:
        with open("context.txt", "r") as file:
            context = file.read()

            if from_api:
                logging.warn("Refreshed settings via API request.")

            return context
    except Exception:
        logging.error("Error reading context.txt, using default.")
        return f"Your name is {assistant_name}. You are the default bot and you are super hyped about it. Considering the following conversation between {user_name} and {assistant_name}, give a single response as {assistant_name}. Do not prefix with your own name. Do not prefix with emojis."

default_context = read_context_file()

def load_model(model_name=current_model_name):
    global current_model_name
    global model
    global tokenizer
    global cache
    global generator
    global streaming_generator

    if model and model_name == current_model_name:
        return

    model_path = fetch_pretrained_model(model_name)

    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    config.max_seq_len = LLM_MAX_SEQ_LEN
    config.scale_pos_emb = LLM_SCALE_POS_EMB
    config.scale_alpha_value = LLM_SCALE_ALPHA

    # Still broken as of ExllamaV2 0.0.11, further research needed
    # LLM_GPU_SPLIT not supported with config.set_low_mem()
    # if LLM_GPU_SPLIT is None:
    #    config.set_low_mem()

    if model:
        logging.warn(f"Unloading {current_model_name} model...")
        model.unload()
        del model

    logging.warn("Loading model: " + model_name)

    model = ExLlamaV2(config, lazy_load=True)
    cache = ExLlamaV2Cache(model, lazy=True)
    model.load_autosplit(cache, LLM_GPU_SPLIT)
    tokenizer = ExLlamaV2Tokenizer(config)
    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
    streaming_generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

    stop_conditions = [tokenizer.eos_token_id] + LLM_STOP_CONDITIONS
    streaming_generator.set_stop_conditions(stop_conditions)

    current_model_name = model_name


def unload():
    global model
    global cache
    global tokenizer
    global generator
    global streaming_generator

    if model is not None:
        logging.warn(f"Unloading {friendly_name}...")
        model.unload()
        del cache
        del model
        del tokenizer
        del generator
        del streaming_generator
        cache = None
        model = None
        tokenizer = None
        generator = None
        streaming_generator = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def offload(for_task: str):
    global friendly_name
    # logging.warn(f"No offload available for {friendly_name}.")
    unload()


async def generate_text(
    prompt: str,
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    temperature: float = 0.7,  # real default is 0.8
    top_k: float = 20,  # real default is 50
    top_p: float = 0.9,  # real default is 0.5
    token_repetition_penalty: float = 1.05,  # real default is 1.05
    typical: float = 1,
) -> AsyncGenerator[str, None]:
    load_gpu_task(friendly_name, Exllama2Client)

    if model is None:
        load_model()

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.token_repetition_penalty = token_repetition_penalty
    settings.typical = typical

    if tokenizer is None:
        raise Exception("tokenizer is NoneType")
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    time_begin = time.time()

    generated_tokens = 0

    input_ids = tokenizer.encode(prompt)
    input_ids.to(autodetect_device())

    streaming_generator.warmup()
    streaming_generator.begin_stream(input_ids, settings, True)

    message = ""

    while True:

        chunk, eos, _ = streaming_generator.stream()
        generated_tokens += 1

        chunk = process_llm_text(chunk, True)

        message = process_llm_text(message + chunk)

        yield chunk

        if eos or generated_tokens + 16 >= max_new_tokens and chunk[-1] in ".?!\n":
            break

    del input_ids

    time_end = time.time()
    time_total = time_end - time_begin

    logging.info(
        f"[exllamav2] Response generated in {time_total:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_total:.2f} tokens/second"
    )


async def chat(
    text: str,
    messages: List[dict],
    context: str = default_context,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    token_repetition_penalty: float = 1.15,
):
    prompt = f"System: {context}\n\n"

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        name = user_name if role == "user" else assistant_name
        prompt += f"\n\n{name}: {content}"

    if text is not None:
        prompt += f"\n\n{user_name}: {text}"

    prompt += f"\n\n{assistant_name}: "

    async_gen = generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token_repetition_penalty=token_repetition_penalty,
        top_p=top_p,
    )

    # combine response to string
    response = ""
    async for chunk in async_gen:
        response += chunk

    return response


async def chat_streaming(
    text: str,
    messages: List[dict],
    context: str = default_context,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p: float = 0.9,
    token_repetition_penalty: float = 1.15,
):
    prompt = f"System: {context}\n\n"

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        name = user_name if role == "user" else assistant_name
        prompt += f"\n\n{name}: {content}"

    if text is not None:
        prompt += f"\n\n{user_name}: {text}"

    prompt += f"\n\n{assistant_name}: "

    async for response in generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        token_repetition_penalty=token_repetition_penalty,
        top_p=top_p,
    ):
        yield response
