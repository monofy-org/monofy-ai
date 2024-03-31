# Note: Major update posted on 3/31/2024, gradio is currently not supported. If you require the old gradio interface, stick with commits from before the rewrite.

# monofy-ai
 Simple and multifaceted API for AI

## What's in the box?
- Python APIs for using large language models, text-to-speech, and Stable Diffusion similarly in your projects
- HTML/Javascript chat interface with image generation and PDF reading abilities, code blocks, chat history, and more
- Gradio interface for experimenting with various features

<img src="https://github.com/monofy-org/monofy-ai/blob/main/github-images/ai-assistant.png" width="300">
<img src="https://github.com/monofy-org/monofy-ai/blob/main/github-images/ai-assistant-sd.png" width="300">

## Requirements
- Windows or Linux, WSL is supported (recommended, even)
- 12GB VRAM (RTX3060 or better recommended)
- 32GB RAM (64GB recommended)
- CUDA 12.1+ (ROCm is currently Linux-only)
- Python 3.10

## Will it run on less than 12GB VRAM?
Your mileage may vary. If you have a lot of CPU ram it may work albeit slowly.

## What models are included automatically?
- OpenOrca Mistral 7B
- Stable Diffusion/XL
- Stable Video Diffusion (img2vid) and ZeroScope (txt2vid)
- Shap-E (3D model generation)
- Coqui/XTTS-v2 (text-to-speech)
- YOLOS (fast object detection)
- moondream (LLM-based object detection)
- AudioGen
- MusicGen
- Whisper (speech-to-text, still experimental)
... and more!

## What additional models are supported?
- EXL2 language models
- Stable Diffusion models (including SDXL and turbo) in .safetensors format

## Can I run everything at the same time?
YES! Don't ask me how, though. It's a secret that you totally won't find by looking in gpu_utils.py.

 ## Why did you make this?
 I just wanted a unified python API for LLM/TTS and possibly even generating simple images. Too many projects require complicated setups, Docker, etc. Many have also become stale or obsolete as huggingface has generously provided improved APIs and examples. Mainly I wanted something simple enough to modify for my exact needs in any scenario without a huge learning curve. I tried to leave everything accessible enough for you to do the same.
 
 ## This project has 3 main goals in mind.

 1. Do what I personally need for my projects (I hope it serves you too!)
 2. No complicated installation steps
 3. Something ready to use, fine-tuned right out of the box

 ## Startup:
 (Note: Some of this is temporary until I decide on a proper way of handling settings.)
 
 A working run.bat is included for reference, but feel free to use your environment of choice (conda, WSL, etc).

 ### The following startup flags are available:
```
 --all .................. Run everything (recommended)
 --api, --webui ......... Launch FastAPI and/or Gradio webui with the following flags
 --tts, --llm, --sd ..... Enable text-to-speech, exllama2, and/or stable diffusion
```

### The following API endpoints are available (please note that this is not a complete list as new features are being added constantly):
Sure, here's the README formatted as a GitHub README.md:

# monofy-ai API

## Image Processing
- `/api/img/canny`
- `/api/img/depth`
- `/api/img/depth/midas`

## Image Generation (text-to-image)
- `/api/txt2img`
- `/api/img2img`
- `/api/inpaint`
- `/api/txt2img/canny`
- `/api/txt2img/depth`
- `/api/txt2img/instantid`
- `/api/txt2img/cascade`
- `/api/txt2img/controlnet`
- `/api/txt2model/avatar`
- `/api/txt2model/avatar/generate`

## Image Generation (image-to-image)
- `/api/img2img`

## 3D Model Generation
- `/api/txt2model/shape`
- `/api/img2model/lgm`
- `/api/img2model/tsr`

## Video Generation (text-to-video)
- `/api/txt2vid/zero`
- `/api/img2vid/xt`
- `/api/txt2vid/animate`
- `/api/txt2vid/zeroscope`

## Image Processing
- `/api/rembg`

## Computer Vision
- `/api/detect/yolos`
- `/api/vision`

## Image-to-Text
- `/api/img2txt/llava`

## Audio
- `/api/musicgen`

## Text Generation
- `/api/chat/completions`
- `/api/chat/stream`
- `/api/txt/summary`
- `/api/txt/profile`

## PDF
- `/api/pdf/rip`

## YouTube Tools
- `/api/youtube/download`
- `/api/youtube/captions`
- `/api/youtube/grid`
- `/api/youtube/frames`

## Text-to-Speech (TTS)
- `/api/tts`

## Other
- `/api/google/trends`

### Adding additional TTS voices
Add wav files containing samples of the voices you want to use into the `voices/` folder. A single example `female1.wav` is included. The `voice` parameter of the tts API expects the name of the file (without .wav on the end). There is no training required!


## Thanks for trying this project! Please file issue reports for feature requests including additional API parameters, etc!
