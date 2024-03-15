import logging
import traceback
import time
from typing import List
import uuid
from fastapi.routing import APIRouter
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from modules.plugins import (
    SupportedPipelines,
    use_plugin,
    get_resource,
    release_plugin,
)
from settings import LLM_MAX_NEW_TOKENS
from utils.gpu_utils import autodetect_device
from utils.text_utils import process_llm_text, remove_emojis
from exllamav2.generator import ExLlamaV2Sampler

router = APIRouter()
device = autodetect_device()
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


class ChatCompletionRequest(BaseModel):
    messages: list
    model: str = "local"
    temperature: float = 0.7
    top_p: float = 0.9
    max_emojis: int = 1  # -1 to disable, 0 = no emojis
    max_tokens: int = LLM_MAX_NEW_TOKENS
    frequency_penalty: float = 1.05
    presence_penalty: float = 0.0
    stream: bool = False


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):

    try:
        content = ""
        token_count = 0

        response = await chat(
            text=None,
            messages=request.messages,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            token_repetition_penalty=request.frequency_penalty,
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
                "completion_tokens": token_count,  # Replace with the actual completion_tokens value
                "total_tokens": token_count,  # Replace with the actual total_tokens value
            },
        )

        # print(response)

        return JSONResponse(content=response_data)

    except Exception as e:
        traceback_info = traceback.format_exc()
        logging.error(f"An error occurred: {e}")
        logging.error(traceback_info)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/llm/refresh")
async def refresh_llm_context():
    read_context_file(True)
    return JSONResponse({"success": True})


@router.get("/api/llm")
async def deprecated_llm_api(prompt: str, context: str = None, stream: bool = False):

    messages = []
    if context:
        messages.append({"role": "system", "content": context})

    if stream:
        return StreamingResponse(chat_streaming(prompt, messages))
    else:
        response = await chat(prompt, messages)
        return response


async def generate_text(
    prompt: str,
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
    temperature: float = 0.7,  # real default is 0.8
    top_k: int = 20,  # real default is 50
    top_p: float = 0.9,  # real default is 0.5
    token_repetition_penalty: float = 1.05,  # real default is 1.05
    typical: float = 1,
):
    await use_plugin(SupportedPipelines.EXLLAMAV2)

    try:
        tokenizer = get_resource(SupportedPipelines.EXLLAMAV2, "tokenizer")
        generator = get_resource(SupportedPipelines.EXLLAMAV2, "generator")

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.token_repetition_penalty = token_repetition_penalty
        settings.typical = typical

        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

        time_begin = time.time()

        generated_tokens = 0

        input_ids = tokenizer.encode(prompt)
        input_ids.to(device)

        print("DEBUG", len(input_ids[0]))

        full_token_count = len(input_ids[0])

        generator.warmup()
        generator.begin_stream(input_ids, settings, True)

        message = ""

        while True:

            chunk, eos, _ = generator.stream()
            generated_tokens += 1
            chunk = process_llm_text(chunk, True)
            message = process_llm_text(message + chunk)

            yield chunk

            if eos or (
                len(chunk) > 0
                and generated_tokens + 16
                >= max_new_tokens  # don't start a sentence with less than 16 tokens left
                and chunk[-1] in ".?!\n"  # finish current sentence even if it's too long
                and message.count("```") % 2 == 0  # finish code block
            ):
                break

        del input_ids

        full_token_count += generated_tokens

        time_total = time.time() - time_begin

        release_plugin(SupportedPipelines.EXLLAMAV2)

        logging.info(
            f"Generated {generated_tokens} tokens, {generated_tokens / time_total:.2f} tokens/second, {full_token_count} total tokens."
        )
    except Exception as e:        
        logging.error(f"An error occurred: {e}", exc_info=True)        
        raise e


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
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
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
