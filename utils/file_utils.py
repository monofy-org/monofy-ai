import logging
import os
import random
import string
import requests
from huggingface_hub import snapshot_download
import torch
from settings import MEDIA_CACHE_DIR
from utils.gpu_utils import autodetect_device, autodetect_dtype
from settings import USE_XFORMERS


def ensure_folder_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created folder {path}")


def import_model(
    model_type,
    repo_or_safetensors: str,
    set_variant_fp16=True,
    allow_fp16=True,
    allow_bf16=True,
    offload=True,
    sequential_offload=False,
    **kwargs,
):
    global pipelines

    single_file = repo_or_safetensors.endswith(".safetensors")

    half = allow_fp16 or allow_bf16

    dtype = autodetect_dtype(allow_bf16) if half else torch.float32

    model_kwargs = {}

    if half:
        model_kwargs["torch_dtype"] = dtype

    logging.info(
        f"Loading {model_type.__name__} from {repo_or_safetensors} {model_kwargs}"
    )

    if single_file:
        if not os.path.exists(repo_or_safetensors):
            raise FileNotFoundError(f"Model not found at {repo_or_safetensors}")

        model: model_type = model_type.from_single_file(
            repo_or_safetensors,
            **model_kwargs,
            **kwargs,
        )

    else:
        if set_variant_fp16:
            model_kwargs["variant"] = "fp16"

        model_path = fetch_pretrained_model(repo_or_safetensors)

        if hasattr(model_type, "from_pretrained"):
            model: model_type = model_type.from_pretrained(
                model_path,
                **model_kwargs,
                **kwargs,
            )
        elif hasattr(model_type, "get_pretrained"):
            model: model_type = model_type.get_pretrained(
                model_path,
                **model_kwargs,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Model {model_type.__name__} has neither from_pretrained nor get_pretrained"
            )

    if torch.cuda.is_available():
        if sequential_offload and hasattr(model, "enable_sequential_cpu_offload"):
            # logging.info(f"Enabling sequential offload for {repo_or_safetensors}")
            model.enable_sequential_cpu_offload()
        elif offload and hasattr(model, "enable_model_cpu_offload"):
            # logging.info(f"Enabling offload for {repo_or_safetensors}")
            model.enable_model_cpu_offload()
        else:
            if not offload:
                logging.warn(f"Offload disabled for model {repo_or_safetensors}")
            else:
                logging.debug(f"Offload unavailable for model {repo_or_safetensors}")

            if hasattr(model, "to"):
                model.to(device=autodetect_device(), dtype=dtype)
            elif hasattr(model, "cuda"):
                model.cuda()

        if USE_XFORMERS:
            if hasattr(model, "enable_xformers_memory_efficient_attention"):
                # logging.info(f"Enabling xformers for {repo_or_safetensors}")
                model.enable_xformers_memory_efficient_attention()
            else:
                logging.debug(f"No xformers available for model {repo_or_safetensors}")

    return model


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

    # Generate a random filename and append the extension
    filename = random_filename(file_extension=extension)

    download_path = os.path.join(MEDIA_CACHE_DIR, filename)
    if not os.path.exists(download_path):
        logging.info(f"Downloading {url} to {download_path}")
        r = requests.get(url, allow_redirects=True)
        open(download_path, "wb").write(r.content)
    return download_path
