import logging
import os
import re
import emoji
import json


# read res/emoji_dict.json
try:
    with open(os.path.join("res", "emoji_dict.json"), "r", encoding="utf-8") as f:
        emoji_dict = json.loads(f.read())
except FileNotFoundError:
    logging.error("res/emoji_dict.json not found")
    emoji_dict = {}
except json.JSONDecodeError:
    logging.error("Error loading emoji_dict.json")
    emoji_dict = {}


def is_emoji(char: str):
    return emoji.emoji_count(char) > 0


def translate_emojis(text: str):
    for emoji_char, replacement in emoji_dict.items():
        if emoji_char:  # check if emoji_char is not empty
            text = text.replace(emoji_char, replacement + " ")
    text = text.replace("  ", " ")
    return text


def remove_emojis(text: str):
    return "".join(char for char in text if not is_emoji(char))


def strip_emojis(text: str):
    # Remove emojis from the beginning and end, and trim whitespace
    return remove_emojis(text).strip()


def json_from_chat(chat: str):
    # get string from first { to last }
    start = chat.find("{")
    end = chat.rfind("}")
    chat = chat[start : end + 1]

    try:
        data = json.loads(chat)
        print(data)
        return data
    except Exception as e:
        print("Error parsing JSON: ", e)


def process_text_for_tts(text: str):

    # remove emotions like *waves* or *gasps* with asterisks around them
    text = re.sub(r"\*.*?\*", "", text)
    text = re.sub(r"[\[\]`“”\"\*;]", "", text)

    return (
        remove_emojis(text)
        .replace("[END]", "")  # remove end markers
        .replace("[END CALL]", "")  # remove end markers
        .replace("[TRANSFER]", "")  # remove end markers
        .replace("[SEARCH]", "")  # remove end markers
        # .replace(",", "")  # commas pause too long by default
        .replace(":", " ")
        .replace("(", ",")
        .replace(")", "")
        .replace(";", ".")  # these need pauses
        .replace(" - ", "- ")  # pauses too long
        .replace("--", "-")  # pauses too long
        .replace("  ", "")
        .replace("  ", "")
        .replace("AI", "A.I.")  # it can't say AI right lol
        .replace("cater", "cayter")  # it loves this word but can't say it for s***
        .replace("macrame", "macra-may")
        .replace("charades", "sharades")
        .replace("\n", "")        
    ).strip() + "."


def close_backquotes(string):
    count = string.count("```")

    if count % 2 == 1:
        return string + "\n```"
    else:
        return string


def process_llm_text(text: str, is_chunk: bool = False):
    text = (
        text.replace(" .", ".")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace("\n``prompt", "\n```prompt")
        .replace("\n```promt", "\n```prompt")
        .replace("```prom\n", "```prompt\n")
    )

    return text if is_chunk else close_backquotes(text)


def csv_to_list(csv_string: str):
    return [x.strip() for x in csv_string.split(",") if x.strip() != ""]
