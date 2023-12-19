import logging
import sys

from settings import DEVICE, USE_DEEPSPEED, USE_XFORMERS


def sys_info():    
    python_info = sys.version
    optimizations = f"xformers={USE_XFORMERS}, deepspeed={USE_DEEPSPEED}"
    logging.info(f"Python version: {python_info}")
    logging.info(f"Using device: {DEVICE} ({optimizations})")