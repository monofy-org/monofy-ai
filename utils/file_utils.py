import logging
import os
import random
import shutil
import string
import requests
import hashlib
from huggingface_hub import snapshot_download
from settings import CACHE_PATH


def ensure_folder_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created folder {path}")


def cached_snapshot(model_name: str):

    user_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    local_dir = (
        user_path + "/models--" + model_name.replace("/", "--").replace(":", "--")
    )

    if os.path.isdir(local_dir):

        if os.path.isdir(os.path.join(local_dir, "snapshots")):
            snapshots_folder = os.path.join(local_dir, "snapshots")
            first_subfolder = os.listdir(snapshots_folder)[0]
            return os.path.abspath(
                os.path.join(local_dir, "snapshots", first_subfolder)
            )

        return local_dir

    logging.info(f"Downloading {model_name} to {local_dir}")

    s = model_name.split(":")
    if len(s) == 2:
        model_name = s[0]
        revision = s[1]
    else:
        revision = "main"

    return snapshot_download(
        repo_id=model_name,
        revision=revision,
        local_dir=local_dir,
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
        filename = os.path.join(CACHE_PATH, filename)
    if file_extension is not None:
        filename += f".{file_extension}"
    return filename


def url_hash(url: str) -> str:
    # Use SHA-256 to generate a unique hash for the URL
    sha256_hash = hashlib.sha256(url.encode()).hexdigest()
    return sha256_hash


def download_to_cache(url: str, extension: str):

    hash = url_hash(url)    
    filename = os.path.join(CACHE_PATH, f"{hash}.{extension}")

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        logging.info(f"Using cached file: {filename}")
    else:
        extension = extension or url.split(".")[-1]
        logging.info(f"Downloading {url} to {filename}")
        r = requests.get(url, allow_redirects=True)
        with open(filename, "wb") as f:
            f.write(r.content)

    return filename
