import os
from huggingface_hub import snapshot_download
import yaml


def get_characters():
    characters = []
    characters_dir = "characters"

    if os.path.exists(characters_dir):
        for filename in os.listdir(characters_dir):
            if filename.endswith(".yaml"):
                characters.append(filename)

    return characters


def get_character(filename: str):

    if not filename:
        raise ValueError("No character filename provided")

    if not isinstance(filename, str):
        print(f"filename={filename}", type(filename))
        raise ValueError("Character filename must be a string")

    path = os.path.normpath(os.path.join("characters", filename))
    if "/" in path or "\\" in path or ".." in path:
        raise Exception("not allowed")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Character file not found: {path}")

    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file.read())

    return yaml_data


def convert_gr_to_openai(chat_list):
    formatted_chat = []

    for user_text, bot_text in chat_list:
        # Append user message
        formatted_chat.append({"role": "user", "content": user_text})

        # Append assistant message
        formatted_chat.append({"role": "assistant", "content": bot_text})

    return formatted_chat


def get_model(model_name: str):
    if model_name.startswith("."):
        model_path = model_name
    else:
        if ":" in model_name:
            revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            model_path = snapshot_download(model_name, revision=revision)
        else:
            model_path = snapshot_download(model_name)

    return model_path
