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
- 12GB VRAM (RTX3060 or Ti4060 recommended)
- 32GB RAM (64GB recommended)
- CUDA 12.1+ (ROCm is currently Linux-only)
- Python 3.10 (may work on 3.11, file an issue if you have any)

## Will it run on less than 12GB VRAM?
Your mileage may vary. If you have a lot of CPU RAM, many features will still work (slowly and/or with lower resolution etc).

## What is included?
- Large language model using Exllamav2
- Stable Diffusion: (SD1.5, SDXL, Turbo, Lightning, Cascade, InstantID supported)
- Video: Stable Video Diffusion, XT, AnimateLCM with multiple interpolation techniques available
- Audio: MusicGen, AudioGen
- Text-to-speech: XTTS with instant voice cloning from 6-20sec samples, edge TTS api also included
- Canny and depth detection with text-to-image IP adapter support
- Vision: YOLOS, Moondream, LLaVA
- Speech dictation using Whisper
- 3D model generation: Shap-E, TripoSR, LGM Mini
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

### The following API endpoints are available (please note that this is not a complete list as new features are being added constantly):

## Image Processing
- `/api/img/canny`
- `/api/img/depth`
- `/api/img/depth/midas`

## Image Generation
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
