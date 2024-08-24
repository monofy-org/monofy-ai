import logging
from fastapi import HTTPException

from classes.requests import Txt2ImgRequest
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS
from utils.text_utils import translate_emojis


def filter_request(req: Txt2ImgRequest):

    req.model_index = req.model_index or SD_DEFAULT_MODEL_INDEX
    model_name = SD_MODELS[req.model_index]
    is_xl = "xl" in model_name.lower()
    is_turbo = "turbo" in model_name.lower()
    is_lightning = "lightning" in model_name.lower()

    if not req.num_inference_steps:
        req.num_inference_steps = 10 if is_lightning else 14 if is_turbo else 20

    scale = 768 if is_xl else 512

    if not req.width:
        req.width = scale
    if not req.height:
        req.height = scale

    aspect_ratio = req.width / req.height

    if req.image:
        if req.width > 1920 or req.height > 1920 or req.width * req.height > 1920 * 1920:        
            if req.width > req.height:
                req.width = 1920
                req.height = int(1920 / aspect_ratio)
            else:
                req.height = 1920
                req.width = int(1920 * aspect_ratio)
        elif req.width < scale or req.height < scale:
            if req.width > req.height:
                req.width = scale
                req.height = int(scale / aspect_ratio)
            else:
                req.height = scale
                req.width = int(scale * aspect_ratio)

    if req.width < 64 or req.height < 64:
        raise HTTPException(400, "Image dimensions should be at least 64x64")

    if req.width % 64 != 0 or req.height % 64 != 0:
        req.width = req.width - (req.width % 64)
        req.height = req.height - (req.height % 64)
        logging.warning(
            f"Image dimensions should be multiples of 64. Cropping to {req.width}x{req.height}"
        )
    if not req.num_inference_steps:
        model_path: str = SD_MODELS[req.model_index]
        req.num_inference_steps = 12 if "xl" in model_path.lower() else 24

    prompt = translate_emojis(req.prompt)
    words = prompt.lower().replace(",", " ").split(" ")

    # Prompts will be rejected if they contain any of these (including partial words).
    # There is honestly no way to block words because the the model understands misspellings.
    # Outgoing images will be checked for age, so this is really just to send a warning.

    banned_partials = [
        "infant",
        "child",
        "toddler",
        "teen",  # Sorry Bruce Springsteen, I'm the boss here.
        "underage",
        "pubesc",
        "minor",
        "school",
        "student",
        "youth",
        "juvenile",
    ]

    # These words are banned only if they are complete words to prevent false positives.
    # For example, "kid" is banned, but "kidney" is not when performing a request that could be nsfw.
    banned_nsfw_words = [
        "kid", "baby", "babies"
    ]

    # Same but more strict (includes partials)
    banned_nsfw_partials = ["boy", "girl"]

    # (partials) These automatically trigger nsfw. I am not going to include a comprehensive
    # list of filthy words because again, the model understands misspellings.
    # Outgoing images are censored if they contain nsfw content (though they may still be nsfw).
    nsfw_partials = [
        "nud",
        "naked",
        "porn",
        "nsfw",
        "sex",
        "boob",
        "breast",
        "penis",
        "vagin",
        "dick",
        "cock",
        "puss",
        "lick",
        "suck",
        "blow",
        "genital",
        "titt",
        "nippl",
    ]


    if req.negative_prompt is None:
        req.negative_prompt = ""
    else:
        req.negative_prompt += ", "

    # force these negative prompts
    req.negative_prompt += "child, teenager, watermark"

    if not req.nsfw:
        req.negative_prompt += ", nudity, nsfw"

    for word in words:

        word = word.lower()

        for banned in banned_partials:
            if banned in word:
                logging.error(prompt)
                raise HTTPException(406, "Prohibited prompt")

        if req.nsfw:
            if word in banned_nsfw_words:
                logging.error(prompt)
                raise HTTPException(406, "Prohibited prompt")
            for banned in banned_nsfw_partials:
                if banned in word:
                    logging.error(prompt)
                    raise HTTPException(406, "Prohibited prompt")

        else:
            for banned in nsfw_partials:
                if banned in word:
                    logging.error(prompt)
                    raise HTTPException(406, "Prohibited prompt")

    return req
