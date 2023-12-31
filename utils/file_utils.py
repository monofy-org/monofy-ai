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


def fetch_pretrained_model(model_name: str, subfolder: str):
    path = f"models/{subfolder}/models--{model_name.replace('/', '--')}"
    if os.path.isdir(path):
        return os.path.abspath(path)
    else:
        return snapshot_download(
            repo_id=model_name,
            cache_dir=os.path.join("models", subfolder),
            local_dir=path,
        )


def delete_file(file_path: str):
    try:
        os.remove(file_path)
        logging.info(f"Deleted {file_path}")
    except Exception as e:
        logging.error(f"Error deleting {file_path}: {e}")


def random_filename(
    file_extension: str = None, include_cache_path=False, length: int = 10
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

    # Generate a random filename and append the extension
    filename = random_filename(file_extension=extension)

    download_path = os.path.join(MEDIA_CACHE_DIR, filename)
    if not os.path.exists(download_path):
        logging.info(f"Downloading {url} to {download_path}")
        r = requests.get(url, allow_redirects=True)
        open(download_path, "wb").write(r.content)
    return download_path
