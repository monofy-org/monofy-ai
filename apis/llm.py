import logging
import traceback
from fastapi import FastAPI
from utils.text_utils import process_llm_text


def llm_api(app: FastAPI):
    import time
    from fastapi import WebSocket, HTTPException
    from fastapi.responses import JSONResponse
    from settings import LLM_MAX_NEW_TOKENS

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict):
        model = body.get("model")
        messages = body.get("messages")
        # stream = True
        temperature = body.get("temperature", 0.7)
        max_sentences = body.get("max_sentences", 3)
        max_tokens = body.get("max_tokens", LLM_MAX_NEW_TOKENS)
        top_p = body.get("top_p", 0.9)
        # frequency_penalty = body.get("frequency_penalty", 1.18)
        from clients import Exllama2Client

        try:
            content = ""
            token_count = 0
            sentence_count = 0

            for chunk in Exllama2Client.chat(
                None,
                messages,
                temperature=temperature,
                max_new_tokens=max_tokens,  # TODO calculate input tokens
                top_p=top_p,
                # frequency_penalty=frequency_penalty,
            ):
                content += chunk
                token_count += 1
                if len(chunk > 0) and chunk[-1] in ".?!":
                    sentence_count += 1
                if sentence_count >= max_sentences:
                    break

            content = process_llm_text(content)

            response_data = {
                "id": "your_id_here",  # Replace with the appropriate ID
                "object": "text_completion",
                "created": int(time.time()),  # Replace with the appropriate timestamp
                "model": model,
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

    @app.get("/api/llm/refresh")
    async def refresh_llm_context():
        from clients import Exllama2Client

        Exllama2Client.read_context_file(True)
        return JSONResponse({"success": True})

    @app.get("/api/llm")
    async def deprecated_llm_api(prompt: str, messages=[], chunk_sentences=True):
        from clients import Exllama2Client

        try:
            content = ""
            for chunk in Exllama2Client.chat(
                prompt, messages, chunk_sentences=chunk_sentences
            ):
                content += chunk

            content = process_llm_text(content)

            response_data = {"choices": [{"message": {"content": content}}]}
            return JSONResponse(content=response_data)

        except Exception as e:
            logging.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/api/llm/stream")
    async def deprecated_llm_websocket(websocket: WebSocket):
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
