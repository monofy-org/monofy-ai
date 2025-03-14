import os
import logging
import asyncio
import sys
import time
from utils.gpu_utils import use_fp16, is_bf16_available, autodetect_device
import torch
from settings import USE_ACCELERATE, USE_DEEPSPEED, USE_XFORMERS


def sys_info():
    python_info = sys.version
    optimizations = f"accelerate={USE_ACCELERATE}, bf16={is_bf16_available}, fp16={use_fp16}, cudnn={torch.backends.cudnn.is_available()}, xformers={USE_XFORMERS}, deepspeed={USE_DEEPSPEED}"
    logging.info(f"Python version: {python_info}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Default device: {autodetect_device()} ({optimizations})")


def show_ram_usage(message="Process memory usage"):
    import psutil

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2
    logging.info(f"{message}: {mem:.2f} MB")


def print_completion_time(since, task_name=None):
    t = time.time() - since
    logging.info(f"{task_name or 'Task completed'} in {round(t,2)} seconds.")
    return t


def sync_generator_wrapper(async_generator):
    loop = asyncio.get_event_loop()
    try:
        while True:
            yield loop.run_until_complete(async_generator.__anext__())
    except StopAsyncIteration:
        pass
    except Exception as e:
        logging.error(f"Error in async generator: {e}")
        raise e
