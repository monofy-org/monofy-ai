import logging
from diffusers.utils.import_utils import is_xformers_available
from utils.gpu_utils import autodetect_device, fp16_available

LOG_LEVEL = logging.INFO

# FastAPI
HOST = "127.0.0.1"
PORT = 5000

# ------------------------
# DEVICE AND OPTIMIZATIONS
# ------------------------
# Can be manually assigned to "cuda:0" etc
DEVICE = autodetect_device()
# Can be set to False to use full 32-bit precision
USE_FP16 = fp16_available()
# By default, xformers and accelerate are used on CUDA (disable for ROCm)
USE_XFORMERS = is_xformers_available()
# Experimental/unused
USE_DEEPSPEED = False  # First, pip install deepspeed (good luck on Windows)

TTS_VOICES_PATH = "voices"
MEDIA_CACHE_DIR = ".cache"

# THIS PLATFORM HAS ONLY BEEN TESTED WITH THESE MODELS
# For LLM, any exl2 model will work but may require adjusting settings
# For TTS, stick to XTTS-v2 or use --edge-tts
# For SD, use hf model tags or the path to a .safetensors file
LLM_MODEL = "LoneStriker/dolphin-2.2.1-mistral-7b-4.0bpw-h6-exl2"  # hf model tag
# LLM_MODEL = "TheBloke/Orca-2-7B-GPTQ" # experimental
TTS_MODEL = "coqui/XTTS-v2"  # hf model tag
SD_MODEL = "runwayml/stable-diffusion-v1-5"
# SD_MODEL = "stabilityai/sdxl-turbo"
# SD_MODEL = "models/sd/realisticVisionV51_v51VAE.safetensors"
# SD_MODEL = "models/sdxl/pixelwaveturbo_01.safetensors" # be sure to set SD_USE_SDXL = True

# Stable Diffusion settings
SD_USE_SDXL = False  # Set to True for SDXL/turbo models
SD_USE_HYPERTILE = True # Use hypertile on images larger than 512px width or height
SD_USE_HYPERTILE_VIDEO = False # Experimental
SD_DEFAULT_STEPS = 20  # Set to 20-40 for non turbo models, or 6-10 for turbo
SD_DEFAULT_WIDTH = 512
SD_DEFAULT_HEIGHT = 512
SD_DEFAULT_GUIDANCE_SCALE = 4.0  # If guidance_scale is not provided (default = 4.0)
SD_USE_VAE = False  # Load separate VAE model

# LLM settings
LLM_DEFAULT_SEED = -1  # Use -1 for a random seed on each reply (recommended)
LLM_MAX_SEQ_LEN = 4096  # Sequence length (default = 4096 but you can go higher with some models)
LLM_MAX_NEW_TOKENS = 80 # Max tokens per response
# (recommended = 1.5-2.0 @ 4096) 1.0 works great but generates lengthy replies
LLM_SCALE_POS_EMB = 2.0
# Split between multiple GPUs, 4000 is enough for the default model
LLM_GPU_SPLIT = None #[4000]

# These values are what appear in chat logs which the model is "completing" on each request
# OpenAI message format will be converted to "Name: message\n\n" and dumped as a single message
LLM_DEFAULT_USER = "User"
LLM_DEFAULT_ASSISTANT = "Assistant"

# Used when determining if a response was cut off
# When chunking by sentences, this can cause emojis at the end to be truncated
# Who cares, though? Plus once they end with an emoji they constant keep doing it
LLM_VALID_ENDINGS = [".", "?", "!", "}", "```"]

# These values are added in addition to the model's built-in eos_token_id value
# No exact science implemented here so feel free to adjust as needed
LLM_STOP_CONDITIONS = [
    "\n\n--",
    "\n\n##",
    f"\n\n{LLM_DEFAULT_USER}:",
    f"\r{LLM_DEFAULT_USER}:",
    f"\n{LLM_DEFAULT_USER}:",
    f"\n\n{LLM_DEFAULT_ASSISTANT}:",
    f"\r{LLM_DEFAULT_ASSISTANT}:",
    f"\n{LLM_DEFAULT_ASSISTANT}:",
    "[img]",
    "(This response",
    "\nRemember, ",
    "[End]"
]
