import os
import torch
# from diffusers.utils.import_utils import is_xformers_available

# Feel free to mess with these:
HOST = "127.0.0.1"
PORT = 5000
CACHE_PATH = ".cache"
TTS_VOICES_PATH = "voices"
TTS_MODEL = "coqui/XTTS-v2:v2.0.2"  # (2.0.3) tends to make male voices sound British
LLM_MODEL = "DrNicefellow/Mistral-Nemo-Instruct-2407-exl2-4bpw"
LLM_MAX_SEQ_LEN = 4096
LLM_SCALE_POS_EMB = LLM_MAX_SEQ_LEN / 4096
LLM_SCALE_ALPHA = 1
LLM_MAX_NEW_TOKENS = 100  # Approximate (sentences are allowed to finish)
SD_DEFAULT_MODEL_INDEX = 0  # Index of the default model in models-sd.txt
TXT2VID_DEFAULT_MODEL_INDEX = 1  # Must be an SD 1.5 model in models-sd.txt
KEEP_FLUX_LOADED = True  # Keep FLUX offloaded (but still loaded in RAM) after first use
AUDIOGEN_MODEL = "facebook/audiogen-medium"
MUSICGEN_MODEL = "facebook/musicgen-stereo-medium"
SVD_MODEL = (
    "stabilityai/stable-video-diffusion-img2vid-xt-1-1"  # requires authentication
)


# Only mess with these if you know what you're doing:
USE_BF16 = True
USE_ACCELERATE = torch.cuda.is_available()  # If True, overrides USE_BF16 and uses it
USE_DEEPSPEED = os.name != "nt"  # Linux/WSL only, improves TTS streaming speed
USE_XFORMERS = False
SD_MIN_IMG2IMG_STEPS = 6  # Minimum steps for img2img after strength is applied
SD_MIN_INPAINT_STEPS = 6  # Minimum steps for inpainting after strength is applied
SD_USE_LIGHTNING_WEIGHTS = False  # Use SDXL Lightning LoRA from ByteDance
SD_DEFAULT_UPSCALE_STRENGTH = 1 if SD_USE_LIGHTNING_WEIGHTS else 0.65
SD_USE_HYPERTILE = False  # Use hypertile for images (experimental)
SD_USE_FREEU = False  # Override with freeu= api parameter
SDXL_USE_REFINER = False  # Override with use_refiner= api parameter
SDXL_REFINER_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
SD_DEFAULT_LORA_STRENGTH = 0.8  # Default strength for LoRAs
SD_DEFAULT_GUIDANCE_SCALE = 4.0  # Override with cfg= api parameter
TXT2VID_DEFAULT_GUIDANCE_SCALE = 1  # Override with cfg= api parameter
IMG2VID_DEFAULT_MOTION_BUCKET = 31  # Override with motion_bucket= api parameter
IMG2VID_DECODE_CHUNK_SIZE = 8  # Uses more VRAM for coherency


# It is strongly recommended not to modify these settings:
SD_HALF_VAE = True  # Use half precision for VAE decode step
IMG2VID_MAX_FRAMES = 24  # Override with num_frames= api parameter
TXT2VID_MAX_FRAMES = 24  # Override with num_frames= api parameter
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
    "|",
    "\n\n--",
    "\n\n##",
    f"\n{LLM_DEFAULT_USER}:",
    f"\n{LLM_DEFAULT_ASSISTANT}:",
    "[END]",
    "[End]",
    "\nSystem",
    "\nsystem",
    "\nAssistant",
    "\nassistant",
    "\nUser",
    "\nuser",
    "\nThe above",
    "(This",
    "\nPlease note",
    "\nIn this ",
    "\nThis concludes",
    "\nNote",
    "**Note",
    "(Note:",
]

# Experimental features, do not enable except for testing
SD_USE_TOKEN_MERGING = False  # Applies tomesd.apply_patch, reduces quality
SD_USE_DEEPCACHE = False
HYPERTILE_VIDEO = False  # Use hypertile for video (experimental)
SD_COMPILE_UNET = False
SD_COMPILE_VAE = False


# These are the default/recommended Stable Diffusion models
# This list is only referenced if models-sd.txt is not present
# If you are trying to edit your models list, look in models-sd.txt
SD_MODELS: list[str] = [
    "Lykon/dreamshaper-xl-v2-turbo/DreamShaperXL_Turbo_v2.safetensors",
    "emilianJR/epiCRealism",
]

