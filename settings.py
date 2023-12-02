import logging

LOG_LEVEL = logging.INFO

# FastAPI
HOST = "127.0.0.1"
PORT = 5000

# THIS BOT HAS ONLY BEEN TESTED WITH THESE MODELS
# For LLM, any exl2 model will work but may require adjusting settings
# For TTS, stick to XTTS-v2 or use --edge-tts
LLM_MODEL = "LoneStriker/dolphin-2.2.1-mistral-7b-4.0bpw-h6-exl2" # hf model tag
#LLM_MODEL = "TheBloke/Orca-2-7B-GPTQ" # experimental
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2" # this is specific to XTTS

SD_MODEL = "models/sd/realisticVisionV51_v51VAE.safetensors" # file path
# "models/sd/realisticVisionV51_v51VAE.safetensors"#  
# "models/sdxl/pixelwaveturbo_01.safetensors" # be sure to set SD_USE_SDXL = True
# "stabilityai/sdxl-turbo" # TODO this line is a placeholder, still need to support loading hf tags
SD_USE_SDXL = False # Set to True for SDXL/turbo models
SD_DEFAULT_STEPS = 20 # Set to 20-30 for non turbo models, or 6-10 for turbo
SD_DEFAULT_GUIDANCE_SCALE = 6.0 # If guidance_scale is not provided (default = 6.0)
SD_USE_MODEL_VAE = True # Use the model as the VAE (for models with baked VAE)

LLM_DEFAULT_SEED = -1   # Use -1 for a random seed on each reply (recommended)
LLM_GPU_SPLIT = [4000]  # Split between multiple GPUs, increase if using a larger model
LLM_MAX_SEQ_LEN = 4096  # Sequence length (default = 4096 but you can go higher)
LLM_SCALE_POS_EMB = 1.5 # (recommended = 2.0 @ 4096) 1.0 works great but generates lengthy replies

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
]

# Support for microsoft/DeepSpeed
# install manually in the venv before enabling (good luck on Windows)
USE_DEEPSPEED = False