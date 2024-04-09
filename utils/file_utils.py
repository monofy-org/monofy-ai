import logging
import os
import random
import string
import requests
from huggingface_hub import snapshot_download
from settings import MEDIA_CACHE_DIR


def ensure_folder_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created folder {path}")


def fetch_pretrained_model(model_name: str):
    # path = f"models/{subfolder}/models--{model_name.replace('/', '--')}"
    path = os.path.join("models", model_name.replace(":", "--"))
    if os.path.isdir(path):
        return path
    else:
        s = model_name.split(":")
        if len(s) == 2:
            model_name = s[0]
            revision = s[1]
        else:
            revision = "main"

        return snapshot_download(
            repo_id=model_name,
            # cache_dir=os.path.join("models", subfolder),
            local_dir=path,
            local_dir_use_symlinks=False,
            revision=revision,
        )


def delete_file(file_path: str):
    try:
        os.remove(file_path)
        logging.info(f"Deleted {file_path}")
    except Exception as e:
        logging.error(f"Error deleting {file_path}: {e}")


def random_filename(
    file_extension: str = None, include_cache_path=True, length: int = 10
):
    filename = "".join(random.choice(string.ascii_letters) for _ in range(length))
    if include_cache_path:
        filename = os.path.join(MEDIA_CACHE_DIR, filename)
    if file_extension is not None:
        filename += f".{file_extension}"
    return filename


def download_to_cache(url: str):
    # Extract the extension from the URL
    extension = url.split(".")[-1]

    url_hash = str(hash(url))
    filename = os.path.join(MEDIA_CACHE_DIR, f"{url_hash}.{extension}")    
    
    if os.path.exists(filename):
        logging.info(f"Using cached file: {filename}")
    else:
        logging.info(f"Downloading {url} to {filename}")
        r = requests.get(url, allow_redirects=True)
        open(filename, "wb").write(r.content)

    return filename
