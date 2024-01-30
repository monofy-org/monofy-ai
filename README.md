# Note: This README is a work in progress as of 1/29/23

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
Text-to-speech:
```
/api/tts?model=<xtts|edge-tts>?text=<str>&voice=<str>&temperature=<float>
```
Chat/text completion (OpenAI compatible):
```
see OpenAI documentation
```
Stable diffusion:
```
/api/txt2img?prompt=<str>&negative_prompt=<str>&guidance_scale=<float>&steps=<int>&upscale=<bool>
```
Shap-E:
```
/api/shape?prompt=<str>&format=<gif|ply|glb>&guidance_scale=<float>
```
AudioGen:
```
/api/audiogen?prompt=<str>&temperature=<float>&cfg_coef=<float>&top_p=<float>
```
MusicGen:
```
/api/musicgen?prompt=<str>&temperature=<float>&cfg_coef=<float>&top_p=<float>
```
YOLOS Object Detection:
```
/api/detect/image_url=<url>
```
moondream Image Description:
```
/api/vision/image_url=<url>&prompt=Describe+the+image
```
Stable Video Diffusion img2vid:
```
/api/img2vid?image_url=<url>&steps=<int>&motion_bucket=<int>&width=<int>&height=<int>&fps=<fps>&frames=<int>&noise=<float>
```
ZeroScope txt2vid:
```
/api/txt2vid?prompt=<str>&steps=<int>&width=<int>&height=<int>
```

### Adding additional TTS voices
Add wav files containing samples of the voices you want to use into the `voices/` folder. A single example `female1.wav` is included. The `voice` parameter of the tts API expects the name of the file (without .wav on the end). There is no training required!


## Thanks for trying this project! Please file issue reports for feature requests including additional API parameters, etc!
