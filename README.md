# Note: This README is a work in progress as of 12/20/23

# monofy-ai
 Simple and multifaceted API for AI

## What's in the box?
- Python APIs for using LLM, TTS, Stable Diffusion similarly in your projects
- HTML/Javascript chat interface with image generation and PDF reading abilities, code blocks, chat history, and more
- Gradio interface for experimenting with various features

## Requirements
- 12GB VRAM (RTX3060 or better recommended)
- Python 3.10

## What models are included automatically?
- OpenOrca Mistral 7B
- Stable Video Diffusion
- Stable Diffusion 1.5
- ConsistencyDecoderVAE
- Coqui/XTTS-v2
- AudioGen
- MusicGen
- Shap-E
- YOLOS
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

### The following API endpoints are available:
Text-to-speech:
```
/api/tts?model=<xtts|edge-tts>?text=<str>&voice=<str>
```
Chat/text completion (OpenAI compatible):
```
see OpenAI documentation
```
Stable diffusion:
```
/api/txt2img?prompt=<str>&negative_prompt=<str>&guidance_scale=<float>&steps=<int>
```
Shap-E:
```
/api/shape?prompt=<str>&format=<gif|ply|glb>&guidance_scale=<float>
```
AudioGen:
```
/api/audiogen?prompt=<str>
```
MusicGen:
```
/api/musicgen?prompt=<str>
```
YOLOS Object Detection:
```
/api/detect/image_url=<url>
```
Stable Video Diffusion:
```
/api/img2vid?image_url=<url>&steps=<int>&motion_bucket=<int>&width=<int>&height=<int>&fps=<fps>&frames=<int>&noise=<float>
```

### Adding additional TTS voices
Add wav files containing samples of the voices you want to use into the `voices/` folder. A single example `female1.wav` is included. The `voice` parameter of the tts API expects the name of the file (without .wav on the end). There is no training required!


## Thanks for trying this project! Please file issue reports for feature requests including additional API parameters, etc!
