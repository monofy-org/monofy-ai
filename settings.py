import os
from diffusers.utils.import_utils import is_xformers_available

# FastAPI
HOST = "127.0.0.1"
PORT = 5000

# Seconds of non-use before considering a model ok to offload, even if it's a small task.
# This is a "soft" offload, meaning the model will still be loaded and will only be
# offloaded once a different or conflicting task it requested.
# Examples of compatible tasks are "tts", "exllamav2", and "stable diffusion".
IDLE_OFFLOAD_TIME = 60

# ------------------------
# DEVICE AND OPTIMIZATIONS
# ------------------------
# By default, xformers and accelerate are used on CUDA (disable for ROCm)
USE_XFORMERS = is_xformers_available()
USE_BF16 = True  # (Experimental) make sure you have a compatible GPU
NO_HALF_VAE = False
USE_DEEPSPEED = os.name != "nt"  # Linux/WSL only, improves TTS streaming speed

TTS_VOICES_PATH = "voices"
MEDIA_CACHE_DIR = ".cache"

# THIS PLATFORM HAS ONLY BEEN TESTED WITH THESE MODELS
# For LLM, any exl2 model will work but may require adjusting settings
# For SD, use the path to a .safetensors file localed in ./models/sd or ./models/sdxl
LLM_MODEL = "LoneStriker/dolphin-2.6-mistral-7b-dpo-laser-4.0bpw-h6-exl2"
# LLM_MODEL = "TheBloke/Orca-2-7B-GPTQ" # experimental
TTS_MODEL = "coqui/XTTS-v2"
DEPTH_MODEL = "DPT_Hybrid"  # DPT_Hybrid, DPT_Large, MiDaS_small supported
MUSICGEN_MODEL = "facebook/musicgen-small"  # facebook/musicgen-small, facebook/musicgen-medium supported

SD_MODEL = "wangqixun/YamerMIX_v8"  # great general purpose model
# SD_MODEL = "SG161222/RealVisXL_V3.0_Turbo/RealVisXL_V3.0_Turbo.safetensors"  # photorealistic

# You can also specify a path to a .safetensors file
# SD_MODEL = "D:\\models\\Stable-diffusion\\realisticVisionV51_v51VAE.safetensors"  # be sure to set SD_USE_SDXL = False
# SD_MODEL = "D:\\models\\Stable-diffusion\\turbovisionxlSuperFastXLBasedOnNew_tvxlV431Bakedvae.safetensors"  # be sure to set SD_USE_SDXL = True
# SD_MODEL = "D:\\models\\Stable-diffusion\\openxlVersion14Human_v14HumanPreference.safetensors"  # be sure to set SD_USE_SDXL = True

# Stable Diffusion settings
SD_FIX_FACES = True  # individually inpaint faces
SD_USE_SDXL = True  # Set to True for SDXL/turbo models
SD_CLIP_SKIP = 0  # Reduce num_hidden_layers in CLIP model (0 = disabled)
SD_USE_HYPERTILE = False  # Use hypertile for images (experimental)
SD_USE_HYPERTILE_VIDEO = False  # Use hypertile for video (experimental)
SD_DEFAULT_STEPS = (
    15 if "turbo" in SD_MODEL else 18 if SD_USE_SDXL else 25
)  # Set to 20-40 for non turbo models, or 6-10 for turbo
SD_DEFAULT_WIDTH = 768 if SD_USE_SDXL else 512
SD_DEFAULT_HEIGHT = 768 if SD_USE_SDXL else 512
SD_DEFAULT_SCHEDULER = (
    "euler_a" if SD_USE_SDXL else "euler"
)  # ddim, euler, euler_a, huen, lms, sde supported
SD_DEFAULT_GUIDANCE_SCALE = 3.0 if SD_USE_SDXL else 4.0  # lower guidance on XL/Turbo
SD_CLIP_SKIP = 1  # Reduce num_hidden_layers in CLIP model
SD_USE_VAE = False  # Use separate vae, currently unimplemented

# Experimental, do not enable
SD_COMPILE_UNET = False
SD_COMPILE_VAE = False

# LLM settings
# LLM_DEFAULT_SEED = -1  # Use -1 for a random seed on each reply (recommended)
LLM_MAX_SEQ_LEN = (
    6144  # Sequence length (default = 4096 but you can go higher with some models)
)
LLM_MAX_NEW_TOKENS = 100  # Max tokens per response
# (recommended = 1.5-2.0 @ 4096) 1.0 works great but generates lengthy replies
LLM_SCALE_POS_EMB = LLM_MAX_SEQ_LEN / 4096
LLM_SCALE_ALPHA = 1.0
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
    f"\n\n{LLM_DEFAULT_USER}:",
    f"\r{LLM_DEFAULT_USER}:",
    f"\n{LLM_DEFAULT_USER}:",
    f"\n\n{LLM_DEFAULT_ASSISTANT}:",
    f"\r{LLM_DEFAULT_ASSISTANT}:",
    f"\n{LLM_DEFAULT_ASSISTANT}:",
    "[img]",
    "The above",
    "(This",
    "\nRemember,",
    "\nNotice",
    "Note",
    "[End]",
]
