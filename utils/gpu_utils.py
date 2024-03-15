import gc
import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from settings import USE_BF16

# torch.set_grad_enabled(False)

if torch.cuda.is_available():
    # torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    #torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark_limit = 0


def clear_gpu_cache():

    # get used vram before clearing
    before = torch.cuda.memory_reserved()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def autodetect_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_idle_offload_time(timeout_seconds: float):
    global idle_offload_time
    idle_offload_time = timeout_seconds


def set_seed(seed: int = -1):
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    random.seed(seed)
    np.random.seed(seed)
    logging.info("Using seed " + str(seed))

    if torch.cuda.is_available():
        # Use CUDA random number generator
        torch.cuda.manual_seed(seed)
    else:
        # Use CPU random number generator
        torch.manual_seed(seed)

    return seed


def bytes_to_gib(bytes_value):
    gib_value = bytes_value / (1024**3)
    return gib_value
