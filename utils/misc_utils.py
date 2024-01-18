import logging
import sys
import time
from utils.gpu_utils import use_fp16, is_bf16_available, autodetect_device
import torch

from settings import USE_DEEPSPEED, USE_XFORMERS


def sys_info():
    python_info = sys.version
    optimizations = f"bf16={is_bf16_available}, fp16={use_fp16}, cudnn={torch.backends.cudnn.is_available()}, xformers={USE_XFORMERS}, deepspeed={USE_DEEPSPEED}"
    logging.info(f"Python version: {python_info}")
    logging.info(f"Using device: {autodetect_device()} ({optimizations})")


def print_completion_time(since, task_name=None):
    t = time.time() - since
    logging.info(f"{task_name or 'Task'} completed in {round(t,2)} seconds.")
    return t
