import logging
import random
from typing import Literal
from fastapi import Depends
from pydantic import BaseModel
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.exllamav2 import ExllamaV2Plugin
from utils.text_utils import json_from_chat


class TxtPersonalityRequest(BaseModel):
    min_age: int = 18
    max_age: int = 80
    gender: Literal["random", "male", "female"] = "random"
    description: str = ""


@PluginBase.router.post("/txt/profile", tags=["Text Generation"])
async def generate_personality(
    req: TxtPersonalityRequest
):
    if req.gender == "random":
        req.gender = random.choice(["male", "female"])

    desc = (" (" + req.description + ")") if req.description else ""

    messages = [
        {
            "role": "system",
            "content": f"Your job is to generate a complete personality profile for a ficticious but realistic {req.gender} between {req.min_age} and {req.max_age} years old{desc}. Use an american name that is not too fake sounding. Your response must be in valid JSON format and all brackets closed. Include only the following properties: name, age, history, employment, appearance, psychological, likes, dislikes, expertise, screen_name. Each property is a string which may be comma-separated at most. screen_name must be alphanumeric and 20 characters. For privacy reasons the screen_name should not include their first or last name. Do not add your own additional properties to the JSON output. At the end of the JSON type [END]",
        }
    ]

    plugin = None

    try:
        plugin: ExllamaV2Plugin = await use_plugin(ExllamaV2Plugin)
        response = await plugin.generate_chat_response(messages=messages, max_new_tokens=1000)
        print(response)
        obj = json_from_chat(response)
        return obj

    except Exception as e:
        logging.error(e, exc_info=True)
        raise e

    finally:
        if plugin is not None:
            release_plugin(plugin)


@PluginBase.router.get("/txt/profile", tags=["Text Generation"])
async def personality_get(req: TxtPersonalityRequest = Depends()):
    return await generate_personality(req)
