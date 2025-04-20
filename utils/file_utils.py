import hashlib
import logging
import os
import random
import string

import requests
from huggingface_hub import snapshot_download

from settings import CACHE_PATH


def ensure_folder_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created folder {path}")


def cached_snapshot(model_name: str, ignore_patterns=[], allow_patterns=[]):
    user_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    local_dir = (
        user_path + "/models--" + model_name.replace("/", "--").replace(":", "--")
    )

    if os.path.isdir(local_dir):
        snapshots_folder = os.path.join(local_dir, "snapshots")
        if os.path.isdir(snapshots_folder):
            first_subfolder = os.listdir(snapshots_folder)[0]
            print("First subfolder:", os.path.join(snapshots_folder, first_subfolder))
            return os.path.abspath(os.path.join(snapshots_folder, first_subfolder))

        return local_dir

    logging.info(f"Downloading {model_name} to {local_dir}")

    s = model_name.split(":")
    if len(s) == 2:
        model_name = s[0]
        revision = s[1]
    else:
        revision = "main"

    kwargs = dict(
        repo_id=model_name,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns

    return snapshot_download(**kwargs)


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


def get_cached_media(url: str, audio_only: bool):
    cached_filename = f"{CACHE_PATH}/{url_hash(url)}.{'mp3' if audio_only else 'mp4'}"
    if os.path.exists(cached_filename):
        return cached_filename
    else:
        return None


def download_to_cache(url: str, extension: str):
    hash = url_hash(url)
    filename = os.path.join(CACHE_PATH, f"{hash}.{extension}")

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        logging.info(f"Using cached file: {filename}")
    else:
        extension = extension or url.split(".")[-1]
        logging.info(f"Downloading {url} to {filename}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers, allow_redirects=True, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        block_size = 8192
        wrote = 0

        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=block_size):
                if chunk:
                    wrote = wrote + len(chunk)
                    f.write(chunk)
                    f.flush()

        if total_size and wrote != total_size:
            os.remove(filename)
            raise Exception(
                f"Downloaded file size mismatch. Expected {total_size}, got {wrote}"
            )

    return filename
