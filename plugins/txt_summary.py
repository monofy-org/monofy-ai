import logging
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException, Depends
from modules.plugins import PluginBase, use_plugin, release_plugin
from plugins.exllamav2 import ExllamaV2Plugin
from settings import LLM_MAX_SEQ_LEN


class SummaryRequest(BaseModel):
    url: str
    prompt: Optional[str] = "Summarize the following text scraped from the web."
    max_response_tokens: Optional[int] = 200


class TxtSummaryPlugin(PluginBase):

    name = "Text Summary"
    description = "Text Summary"
    instance = None

    def __init__(self):
        super().__init__()

    async def summarize(self, req: SummaryRequest):
        # download html from url
        import requests

        # download html from url
        response = requests.get(req.url)

        # extract text from html
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()

        while "\n\n" in text:
            text = text.replace("\n\n", "\n")

        llm: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin)        

        print(text)
        logging.info("Generating summary...")

        # tokenize input
        from exllamav2 import ExLlamaV2Tokenizer

        tokenizer: ExLlamaV2Tokenizer = llm.resources["tokenizer"]

        tokens = tokenizer.encode(text)

        count = tokens[0].shape[0]

        words = " ".split(text)

        limit = LLM_MAX_SEQ_LEN - req.max_response_tokens - 100
        while count > limit:
            words.pop(-1)
            tokens = tokenizer.encode(" ".join(words))
            count = tokens[0].shape[0]

        if count > limit:
            # resize tensor
            text = ""
            count = 0
            for token in tokens[0]:
                text += tokenizer.decode(token)
                count += 1
                if count > limit:
                    break

        text = tokenizer.decode(tokens[0])

        response = "".join(
            [
                x
                async for x in llm.generate_chat_response(
                    context="Your job is to summarize content from the web. and answer questions for the user. Please forgive the formatting and the inclusion of text scraped in error. The text is as follows:",
                    messages=[
                        {
                            "role": "system",
                            "content": f"{text}\n\n[INST]Answer the following question about the article. Remember to keep it factual and unopinionated. Do not make up any details. Use less than {limit} words. Do not make up any details not specifically stated.[INST]",
                        },
                        {
                            "role": "user",
                            "content": f"{req.prompt}",
                        }
                    ],
                    max_new_tokens=req.max_response_tokens,
                )
            ]
        )

        release_plugin(ExllamaV2Plugin)

        print(response)

        return {
            "text": text,
            "summary": response,
        }


@PluginBase.router.post("/txt/summary", tags=["Text Generation"])
async def txt_summary(req: SummaryRequest):
    try:
        plugin: TxtSummaryPlugin = await use_plugin(TxtSummaryPlugin, True)
        response = await plugin.summarize(req)
        return JSONResponse(response)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@PluginBase.router.get("/txt/summary", tags=["Text Generation"])
async def txt_summary_from_url(
    req: SummaryRequest = Depends(),
):
    return await txt_summary(req)
