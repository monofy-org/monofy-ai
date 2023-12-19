import logging
import torch
import numpy as np


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


current_task_name: str = None
current_exllamav2 = None
likes_llamas = ["exllamav2", "tts", "stable diffusion"]


def free_vram(for_task_name: str, exllamav2_client=None):
    if torch.cuda.is_available():
        global current_task_name
        global likes_llamas
        if for_task_name != current_task_name:
            if for_task_name == "exllamav2":
                global current_exllamav2
                current_exllamav2 = exllamav2_client
            elif current_exllamav2:
                if for_task_name not in likes_llamas:
                    current_exllamav2.model.unload()
                    del current_exllamav2.model
                    current_exllamav2.model = None

            if (current_task_name in likes_llamas) and (for_task_name in likes_llamas):
                pass

            else:
                before = torch.cuda.memory_reserved()
                torch.cuda.empty_cache()
                after = torch.cuda.memory_reserved()
                gib = bytes_to_gib(before - after)
                if gib > 0:
                    logging.info(f"Freed {round(gib,2)} GiB from VRAM cache")
                logging.info(f"Loading {for_task_name}...")

            current_task_name = for_task_name
