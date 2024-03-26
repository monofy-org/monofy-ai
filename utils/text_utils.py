import emoji
import json


def is_emoji(char: str):
    return emoji.emoji_count(char) > 0


def remove_emojis(text: str):
    return "".join(char for char in text if not is_emoji(char))


def strip_emojis(text: str):
    # Remove emojis from the beginning and end, and trim whitespace
    return remove_emojis(text).strip()


def json_from_chat(chat: str):
    # get string from first { to last }
    start = chat.find("{")
    end = chat.rfind("}")
    chat = chat[start:end + 1]

    try:
        data = json.loads(chat)
        print(data)
        return data
    except Exception as e:
        print("Error parsing JSON: ", e)


def process_text_for_tts(text: str):
    return (
        remove_emojis(text)
        .replace("`", "")  # escape backquotes are common and pointless
        .replace('"', "")  # quotes freak it out
        .replace("“", "")
        .replace("”", "")
        # .replace(",", "")  # commas pause too long by default
        .replace("*", "")  # these are no good
        .replace(":", ".").replace(";", ".")  # these need pauses
        .replace(" - ", "- ")  # pauses too long
        .replace("--", "-")  # pauses too long
        .replace("  ", "")
        .replace("  ", "")
        .replace("AI", "A.I.")  # it can't say AI right lol
        .replace("cater", "cayter")  # it loves this word but can't say it for s***
    ).strip() + " ..."  # add silence to end to prevent early truncation


def close_backquotes(string):
    count = string.count("```")

    if count % 2 == 1:
        return string + "\n```"
    else:
        return string


def process_llm_text(text: str, is_chunk: bool = False):
    text = (
        text.replace(" .", ".")
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
