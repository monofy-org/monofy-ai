import asyncio
import logging
import torch
import numpy as np

gpu_thread_lock = asyncio.Lock()


def autodetect_device():
    """Returns a device such as "cpu" or "cuda:0" """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_seed(seed: int = -1):
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Use CUDA random number generator
        generator = torch.cuda.manual_seed(seed)
    else:
        # Use CPU random number generator
        generator = torch.manual_seed(seed)

    return generator


def bytes_to_gib(bytes_value):
    gib_value = bytes_value / (1024**3)
    return gib_value


def fp16_available():
    try:
        # Attempt to create a NumPy array with dtype=float16
        np.array(1, dtype=np.float16)
        return True
    except TypeError:
        return False


current_tasks = {}
last_used = {}
last_task = None
small_tasks = ["exllamav2", "tts", "stable diffusion"]
large_tasks = ["sdxl", "svd", "shap-e", "audiogen", "musicgen"]


def free_vram(task_name: str, client):
    global current_tasks
    global last_task

    if not torch.cuda.is_available() or task_name in current_tasks:
        return

    small_tasks_only = last_task is not None and last_task in small_tasks

    empty_cache = task_name != last_task

    if small_tasks_only:
        if task_name in large_tasks:
            empty_cache = True
            for _, client in current_tasks.items():
                client.offload(task_name)
            current_tasks.clear()
    else:
        if current_tasks:
            empty_cache = True
            for _, client in current_tasks.items():
                client.offload(task_name)
            current_tasks.clear()

    current_tasks[task_name] = client
    last_task = task_name

    if empty_cache:
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved()
        gib = bytes_to_gib(before - after)
        if gib > 0:
            logging.info(f"Freed {round(gib,2)} GiB from VRAM cache")
        logging.info(f"Loading {task_name}...")
