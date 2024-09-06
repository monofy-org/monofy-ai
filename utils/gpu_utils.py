import gc
import logging
import random
import sys
import time
from typing import Optional, Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from settings import USE_ACCELERATE, USE_BF16
import accelerate
from accelerate import Accelerator

accelerator = Accelerator()
tensor_to_timer = 0

if torch.cuda.is_available():
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.benchmark_limit = 0


def clear_gpu_cache():

    # get used vram before clearing
    before = torch.cuda.memory_reserved()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.cuda.synchronize()

    # get used vram after clearing
    after = torch.cuda.memory_reserved()

    logging.info(
        f"Cleared VRAM: {bytes_to_gib(before-after):.2f} GiB released. {bytes_to_gib(after):.2f} GiB used."
    )


def _check_bf16():
    if not torch.cuda.is_available() or not USE_BF16:
        return False
    try:
        r = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.bfloat16)
        F.interpolate(r, size=(64, 64), mode="nearest")
        return True
    except Exception:
        return False


def _check_fp16():
    if not torch.cuda.is_available():
        return False
    try:
        np.array(1, dtype=np.float16)
        return True
    except TypeError:
        return False


is_bf16_available = _check_bf16()
use_fp16 = _check_fp16()


def autodetect_dtype(bf16_allowed: bool = True):
    if USE_BF16 and bf16_allowed and is_bf16_available:
        return torch.bfloat16
    else:
        return torch.float16 if use_fp16 else torch.float32


def autodetect_variant():
    return "fp16" if use_fp16 else None


def autodetect_device(allow_accelerate: bool = True):
    if USE_ACCELERATE and allow_accelerate:
        return accelerator.device
    elif sys.platform == "darwin" and torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def random_seed_number():
    return random.randint(0, 2**32 - 1)


def set_seed(seed: int = -1, return_generator=False):
    if seed == -1:
        seed = random_seed_number()

    logging.info("Using seed " + str(seed))

    random.seed(seed)
    np.random.seed(seed)

    generator = torch.manual_seed(seed)

    if torch.cuda.is_available():
        # Use CUDA random number generator
        torch.cuda.manual_seed(seed)

    if return_generator:
        return seed, generator

    return seed


def bytes_to_gib(bytes_value):
    gib_value = bytes_value / (1024**3)
    return gib_value


def check_device_same(d1, d2):
    if d1.type != d2.type:
        return False
    if d1.type == "cuda" and d1.index is None:
        d1 = torch.device("cuda", index=0)
    if d2.type == "cuda" and d2.index is None:
        d2 = torch.device("cuda", index=0)
    return d1 == d2


# Directly load to GPU
# credit to vladmandic for this! https://github.com/vladmandic/automatic/
# called for every item in state_dict by diffusers during model load
def hijack_set_module_tensor(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,  # pylint: disable=unused-argument
    fp16_statistics: Optional[
        torch.HalfTensor
    ] = None,  # pylint: disable=unused-argument
):
    global tensor_to_timer  # pylint: disable=global-statement
    if device == "cpu":  # override to load directly to gpu
        device = autodetect_device()
    t0 = time.time()
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            module = getattr(module, split)
        tensor_name = splits[-1]
    old_value = getattr(module, tensor_name)
    with torch.no_grad():
        # note: majority of time is spent on .to(old_value.dtype)
        if tensor_name in module._buffers:  # pylint: disable=protected-access
            module._buffers[tensor_name] = value.to(
                device, old_value.dtype, non_blocking=True
            )  # pylint: disable=protected-access
        elif value is not None or not check_device_same(
            torch.device(device), module._parameters[tensor_name].device
        ):  # pylint: disable=protected-access
            param_cls = type(
                module._parameters[tensor_name]
            )  # pylint: disable=protected-access
            module._parameters[tensor_name] = param_cls(
                value, requires_grad=old_value.requires_grad
            ).to(
                device, old_value.dtype, non_blocking=True
            )  # pylint: disable=protected-access
    t1 = time.time()
    tensor_to_timer += t1 - t0


original_tensor_to_device = accelerate.utils.set_module_tensor_to_device


def enable_hot_loading():
    accelerate.utils.set_module_tensor_to_device = hijack_set_module_tensor


def disable_hot_loading():
    accelerate.utils.set_module_tensor_to_device = original_tensor_to_device
