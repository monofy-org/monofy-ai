import os
from diffusers.utils.import_utils import is_xformers_available

# FastAPI
HOST = "127.0.0.1"
PORT = 5000

# ------------------------
# DEVICE AND OPTIMIZATIONS
# ------------------------
# By default, xformers and accelerate are used on CUDA (disable for ROCm)
USE_XFORMERS = is_xformers_available()
USE_BF16 = True
NO_HALF_VAE = False
USE_DEEPSPEED = os.name != "nt"  # Linux/WSL only, improves TTS streaming speed

TTS_VOICES_PATH = "voices"
MEDIA_CACHE_DIR = ".cache"

# For LLM, any exl2 model will work but may require adjusting settings
LLM_MODEL = "bartowski/Lexi-Llama-3-8B-Uncensored-exl2:4_25"
# LLM_MODEL = "bartowski/dolphin-2.9-llama3-8b-exl2:4_25"
# LLM_MODEL = "bartowski/dolphin-2.8-mistral-7b-v02-exl2:4_25"
# LLM_MODEL = "bartowski/dolphin-2.8-mistral-7b-v02-exl2:3_5"
# LLM_MODEL = "bartowski/laser-dolphin-mixtral-2x7b-dpo-exl2:3_5"

# The latest version (2.0.3) tends to make male voices sound British
TTS_MODEL = "coqui/XTTS-v2:v2.0.2"

AUDIOGEN_MODEL = "facebook/audiogen-medium"  # there is no small version of audiogen
MUSICGEN_MODEL = "facebook/musicgen-small"  # other versions of musicgen should work fine

# Use without -1-1 if you prefer not to authenticate to download the model
SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"

# These are the default/recommended Stable Diffusion models
SD_MODELS = [
    "Lykon/dreamshaper-xl-v2-turbo/DreamShaperXL_Turbo_v2.safetensors",
    "SG161222/RealVisXL_V3.0_Turbo/RealVisXL_V3.0_Turbo.safetensors",  # more photorealistic 
]

# Grab additional model paths from models-sd.txt
if os.path.exists("models-sd.txt"):
    with open("models-sd.txt", "r") as f:
        SD_MODELS = SD_MODELS + f.read().splitlines()

# "D:\\models\\Stable-diffusion\\realisticVisionV51_v51VAE.safetensors"  # be sure to set SD_USE_SDXL = False

SD_DEFAULT_MODEL_INDEX = 0  # Index of the default model in the SD_MODELS list

# Stable Diffusion settings
SD_USE_SDXL = True  # Set to True for SDXL/turbo models
SD_HALF_VAE = True  # Use half precision for VAE decode step
SD_USE_TOKEN_MERGING = False  # Applies tomesd.apply_patch, reduces quality
SD_USE_DEEPCACHE = False
SD_USE_FREEU = False  # Use FreeU for images by default (can be overridden with the freeu= api parameter)
SD_USE_HYPERTILE = False  # Use hypertile for images (experimental)
SD_USE_LIGHTNING_WEIGHTS = False  # Use SDXL Lightning LoRA from ByteDance (fuses on model load, breaks face inpainting)
HYPERTILE_VIDEO = False  # Use hypertile for video (experimental)
SD_DEFAULT_STEPS = (
    8
    if SD_USE_LIGHTNING_WEIGHTS
    else 14 if "turbo" in SD_MODELS[0] else 18 if SD_USE_SDXL else 25
)  # Set to 20-40 for non turbo models, or 6-10 for turbo
SD_DEFAULT_WIDTH = 768 if SD_USE_SDXL else 512
SD_DEFAULT_HEIGHT = 768 if SD_USE_SDXL else 512
SD_DEFAULT_GUIDANCE_SCALE = (
    0 if SD_USE_LIGHTNING_WEIGHTS else 3.0 if SD_USE_SDXL else 4.0
)  # lower guidance on XL/Turbo
SD_DEFAULT_UPSCALE_STRENGTH = 1 if SD_USE_LIGHTNING_WEIGHTS else 0.65
SD_USE_VAE = False  # Use separate vae, currently unimplemented

# Experimental, do not enable
SD_COMPILE_UNET = False
SD_COMPILE_VAE = False

TXT2VID_MAX_FRAMES = 25
IMG2VID_DECODE_CHUNK_SIZE = 3
IMG2VID_DEFAULT_FRAMES = 24
IMG2VID_DEFAULT_MOTION_BUCKET = 31

# LLM settings
# LLM_DEFAULT_SEED = -1  # Use -1 for a random seed on each reply (recommended)
LLM_MAX_SEQ_LEN = (
    6144  # Sequence length (default = 4096 but you can go higher with some models)
)
LLM_MAX_NEW_TOKENS = (
    200  # Approx. max tokens per response (sentences are allowed to finish)
)
LLM_SCALE_POS_EMB = LLM_MAX_SEQ_LEN / 4096
LLM_SCALE_ALPHA = 1.1
# Split between multiple GPUs, 4000 is enough for the default model
LLM_GPU_SPLIT = None  # [4000]

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
    f"\n{LLM_DEFAULT_USER}:",        
    f"\n{LLM_DEFAULT_ASSISTANT}:",
    "[img]",
    "\nThe above",
    "(This",
    "\nPlease note",
    "\nThis conversation",
    "\nIn this ",
    "\nRemember",
    "\nNotice",
    "\nThis concludes",
    "\nNote",
    "(Note:",
    "[END]",
    "[End]",
]
