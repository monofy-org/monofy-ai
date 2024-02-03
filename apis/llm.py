import logging
import traceback
from fastapi.routing import APIRouter
import time
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from settings import LLM_MAX_NEW_TOKENS
from utils.text_utils import process_llm_text
import uuid

router = APIRouter()


class CompletionRequest(BaseModel):
    messages: list
    model: str = "local"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = LLM_MAX_NEW_TOKENS
    max_sentences: int = 3
    frequency_penalty: float = 1.05
    presence_penalty: float = 0.0
    stream: bool = False


@router.post("/v1/chat/completions")
async def chat_completions(request: CompletionRequest):
    from clients import Exllama2Client

    try:
        content = ""
        token_count = 0
        sentence_count = 0

        for chunk in Exllama2Client.chat(
            None,
            request.messages,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,  # TODO calculate input tokens
            top_p=request.top_p,
            token_repetition_penalty=request.frequency_penalty,
        ):
            content += chunk
            token_count += 1
            if len(chunk) > 0 and chunk[-1] in ".?!":
                sentence_count += 1
            if sentence_count >= request.max_sentences:
                break

        content = process_llm_text(content)

        response_data = {
            "id": uuid.uuid4().hex,
            "object": "text_completion",
            "created": int(time.time()),  # Replace with the appropriate timestamp
            "model": request.model,
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # Replace with the actual prompt_tokens value
                "completion_tokens": token_count,  # Replace with the actual completion_tokens value
                "total_tokens": token_count,  # Replace with the actual total_tokens value
            },
        }

        # print(response)

        return JSONResponse(content=response_data)

    except Exception as e:
        traceback_info = traceback.format_exc()
        logging.error(f"An error occurred: {e}")
        logging.error(traceback_info)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/llm/refresh")
async def refresh_llm_context():
    from clients import Exllama2Client

    Exllama2Client.read_context_file(True)
    return JSONResponse({"success": True})


@router.get("/api/llm")
async def deprecated_llm_api(prompt: str, messages=[]):
    from clients import Exllama2Client

    try:
        content = ""
        for chunk in Exllama2Client.chat(prompt, messages):
            content += chunk

        content = process_llm_text(content)

        response_data = {"choices": [{"message": {"content": content}}]}
        return JSONResponse(content=response_data)

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
