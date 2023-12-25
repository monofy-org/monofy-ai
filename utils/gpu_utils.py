import logging
import threading
from regex import I
import torch
import numpy as np
from clients.diffusers.SDClient import SDClient
from clients.llm.Exllama2Client import Exllama2Client

from clients.tts.TTSClient import TTSClient

gpu_thread_lock = threading.Lock()


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


current_tasks: list[str] = []
current_exllamav2 = None
small_tasks = ["exllamav2", "tts", "stable diffusion"]
large_tasks = ["svd"]


def free_vram(task_name: str):
    global current_tasks
    if not torch.cuda.is_available() or task_name in current_tasks:
        return

    global small_tasks
    # First element is a small tasks. We are doing small tasks.
    small_tasks_only = len(current_tasks) > 0 and current_tasks[0] in small_tasks

    if small_tasks_only and task_name in large_tasks:
        # Unload small tasks to make room for a large one
        if "exllamav2" in current_tasks:
            Exllama2Client.instance.offload()
        if "tts" in current_tasks:
            TTSClient.instance.offload()
        if "stable diffusion" in current_tasks:
            SDClient.instance.image_pipeline.enable_model_cpu_offload()

    elif not small_tasks_only and task_name in large_tasks:
        SDClient.instance.video_pipeline.enable_model_cpu_offload()

    else:
        return

    before = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved()
    gib = bytes_to_gib(before - after)
    if gib > 0:
        logging.info(f"Freed {round(gib,2)} GiB from VRAM cache")
    logging.info(f"Loading {task_name}...")

    current_tasks = []
