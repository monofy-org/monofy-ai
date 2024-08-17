import logging
import os
import re
import time
import uuid
import emoji
import json

import yaml

from settings import LLM_DEFAULT_ASSISTANT


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


def generate_combinations(text):
    # Find all lists in the text
    lists = re.findall(r"\[(.*?)\]", text)

    if not lists:
        return [text.strip()]

    # Split the found lists
    lists = [lst.split(",") for lst in lists]

    # Get the maximum length of the lists
    max_length = max(len(lst) for lst in lists)

    # Create combinations
    combinations = []
    for i in range(max_length):
        combination = text
        for lst in lists:
            if len(lst) > i:
                combination = combination.replace(
                    f"[{','.join(lst)}]", lst[i].strip(), 1
                )
            else:
                # Repeat the last item if the list is shorter
                combination = combination.replace(
                    f"[{','.join(lst)}]", lst[-1].strip(), 1
                )
        combinations.append(combination.strip())

    return combinations


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


def get_chat_context(
    messages: list[dict], user_name: str, bot_name: str, context: str = None
):

    if context and context.endswith(".yaml"):
        logging.info(f"Using character: {context}")
        path = os.path.join("characters", context)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # read from characters folder
        with open(path, "r") as file:
            yaml_data = yaml.safe_load(file.read())

        if not bot_name:
            bot_name = yaml_data.get("name", LLM_DEFAULT_ASSISTANT)

        context = yaml_data.get("context", context)
        if not context:
            logging.error("Invalid character file (missing context field)")

    if not bot_name:
        bot_name = LLM_DEFAULT_ASSISTANT

    if not context:
        logging.warn("No context provided, using default.")
        context = "You are {bot_name}, a helpful chat assistant."

    prompt = f"<s>{context}\n\nDigest the following and wait for further instructions.\n\n"

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        name = user_name if role == "user" else bot_name
        prompt += f"\n\n{name}: {content}[END]"

    prompt += f"\n\n[INST] Give a single response as {bot_name}, remembering to use backquotes if your response is a prompt or code block. Do not instruct others on backquotes, commands, or formatting. Do not include quotation marks when talking normally.[/INST]\n</s>\n\nAssistant: "

    prompt = (
        prompt.replace("{bot_name}", bot_name)
        .replace("{name}", bot_name)
        .replace("{user_name}", user_name)
        .replace("{timestamp}", time.strftime("%A, %B %d, %Y %I:%M %p"))
    )

    return prompt


def format_chat_response(content, model, prompt_tokens, completion_tokens):

    return dict(
        id=uuid.uuid4().hex,
        object="text_completion",
        created=int(time.time()),  # Replace with the appropriate timestamp
        model=model,
        choices=[
            {
                "message": {"role": "assistant", "content": content},
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def detect_end_of_sentence(chunk: str):
    return (
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
