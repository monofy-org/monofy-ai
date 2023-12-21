import logging
import sys

import torch

from settings import DEVICE, USE_DEEPSPEED, USE_XFORMERS, USE_FP16


def sys_info():
    python_info = sys.version
    optimizations = (
        f"fp16={USE_FP16}, cudnn={torch.backends.cudnn.is_available()}, xformers={USE_XFORMERS}, deepspeed={USE_DEEPSPEED}"
    )
    logging.info(f"Python version: {python_info}")
    logging.info(f"Using device: {DEVICE} ({optimizations})")
