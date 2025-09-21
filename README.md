# monofy-ai
 Simple and multifaceted API for AI

## What's in the box?
- Python APIs for using large language models, text-to-speech, and Stable Diffusion similarly in your projects
- HTML/Javascript chat interface with image generation and PDF reading abilities, code blocks, chat history, and more
- Gradio interface for experimenting with various features

<img src="https://github.com/monofy-org/monofy-ai/blob/main/github-images/webui-txt2img.png" width="300">
<img src="https://github.com/monofy-org/monofy-ai/blob/main/github-images/ai-assistant.png" width="300">
<img src="https://github.com/monofy-org/monofy-ai/blob/main/github-images/ai-assistant-sd.png" width="300">

## Requirements
- Windows or Linux, WSL is supported (recommended, even)
- 12GB VRAM (RTX3060 or Ti4060 recommended)
- 32GB RAM (64GB recommended)
- Python 3.11
- CUDA 12.4 Toolkit

## Will it run on less than 12GB VRAM?
Your mileage may vary. If you have a lot of CPU RAM, many features will still work (slowly and/or with lower resolution etc).

## What is included?
- Large language model using ExllamaV2 (dolphin-2.8-mistral-7b-v02 by default, other options available)
- Vision: YOLOS, Moondream, Owl, LLaVA, DepthAnything, Midas, Canny, and more
- Speech dictation using Whisper
- Image Generation: (SD1.5, SDXL, SD3/3.5, Turbo, Lightning, Cascade, IC Relight, Flux, and more)
- Video: Stable Video Diffusion XT, LivePortrait, AnimateLCM with multiple modes available
- Audio: ACE-Step, Stable Audio, MusicGen, MMAudio
- Text-to-speech: XTTS with instant voice cloning from 6-20sec samples, edge TTS api also included
- 3D model generation: Hunyuan3D 2, Shap-E, TripoSR, LGM Mini
- Endpoints with combinations of features to automate workflow
- Easy plugin system that copilot understands (write plugins for new HF models in minutes or seconds)
... and much more!

## Are all of these features available out of the box?
Yes! Models and other resources are downloaded automatically. This project aims to fully to utilize the Hugging Face cache system.

 ## Why did you make this?
 I just wanted a unified python API for LLM/TTS and possibly even generating simple images. Too many projects require complicated setups, Docker, etc. Many have also become stale or obsolete as huggingface has generously provided improved APIs and examples. Mainly I wanted something simple enough to modify for my exact needs in any scenario without a huge learning curve. I tried to leave everything accessible enough for you to do the same.
 
 ## This project has 3 main goals in mind.

 1. Do what I personally need for my projects (I hope it serves you too!)
 2. No complicated installation steps
 3. Something ready to use, fine-tuned right out of the box

 ## Startup:
 (Note: Some of this is temporary until I decide on a proper way of handling settings.)
 
 A working run.bat is included for reference, but feel free to use your environment of choice (conda, WSL, etc).

## Linux/WSL pre-configuration:

### Add the deadsnakes PPA (Personal Package Archive)
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
```

### Install Python 3.11 and venv
```
sudo apt install python3.11 python3.11-dev python3.11-venv
```

### The following API endpoints are available (please note that this is not a complete list as new features are being added constantly):

### Image Processing
- `/img/canny`
- `/img/depth`
- `/img/depth/midas`
- `/img/rembg`
- `/vid2densepose`

### Image Generation
- `/txt2img`
- `/img2img`
- `/inpaint`
- `/txt2img/flux`
- `/txt2img/canny`
- `/txt2img/depth`
- `/txt2img/openpose`
- `/txt2img/relight`
- `/txt2img/instantid`
- `/txt2img/cascade`
- `/txt2img/controlnet`

### 3D Model Generation
- `/hy3dgen`
- `/txt2model/shape`
- `/img2model/lgm`
- `/img2model/tsr`

## Video Generation
- `/img2vid/xt`
- `/txt2vid/animate`
- `/txt2vid/zero`
- `/txt2vid/zeroscope`
- `/img2vid/liveportrait`

### Computer Vision
- `/detect/yolos`
- `/vision`

### Image-to-Text
- `/img2txt/llava`

### Audio
- `/txt2wav/ace-step`
- `/txt2wav/stable-audio`
- `/txt2wav/musicgen`
- `/mmaudio`
- `/piano2midi`

### Text Generation
- `/chat/completions`
- `/chat/stream`
- `/txt/summary`
- `/txt/profile`

### YouTube Tools
- `/youtube/download`
- `/youtube/captions`
- `/youtube/grid`
- `/youtube/frames`

### Reddit Tools
- `/reddit/download`

### Text-to-Speech (TTS)
- `/tts`

### Other
- `/google/trends`

### Adding additional TTS voices
Add wav files containing samples of the voices you want to use into the `voices/` folder. A single example `female1.wav` is included. The `voice` parameter of the tts API expects the name of the file (without .wav on the end). There is no training required!


## Thanks for trying this project! Please file issue reports for feature requests including additional API parameters, etc!
