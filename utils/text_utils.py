import emoji


def is_emoji(char: str):
    return emoji.emoji_count(char) > 0


def remove_emojis(text: str):
    return "".join(char for char in text if not is_emoji(char))


def strip_emojis(text: str):
    # Remove emojis from the beginning and end, and trim whitespace
    return remove_emojis(text).strip()


def process_text_for_tts(text: str):
    return (
        remove_emojis(text)
        .replace("`", "")  # escape backquotes are common and pointless
        .replace('"', "")  # regular quotes freak it out
        #.replace(",", "")  # commas pause too long by default
        .replace("*", "")  # these are no good
        .replace(":", "-")  # unnecessary pause?
        .replace(" - ", "- ")  # pauses too long
        .replace("--", "-")  # pauses too long
        .replace("  ", "")  # weird noise
        .replace("AI", "A.I.")  # it can't say AI right lol
        .replace("cater", "cayter")  # it loves this word but can't say it for s***
    ).strip() + " ..."  # add silence to end to prevent early truncation


def process_llm_text(text: str):
    return (
        text.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",")
    )
