# Note: This README is a work in progress

# monofy-ai
 Simple and multifaceted API for AI chat assistance

## What's in the box?
- HTML/Javascript chat interface with image generation and PDF reading abilities, code blocks, chat history, and more
- APIs for using LLM, TTS, Stable Diffusion similarly in your projects
- Gradio interface for experimenting with various features

## What models are supported?
- EXL2 language models
- Stable Diffusion models (1.5, SDXL, turbo) in .safetensors format
- Coqui TTS (model installs automatically) and edge-tts (free through Microsoft)
  
 ## Why did you make this?
 I just wanted a unified python API for LLM/TTS and possibly even generating simple images. Too many projects require complicated setups, Docker, etc. Many have also become stale or obsolete as huggingface has generously provided improved APIs and examples. Mainly I wanted something simple enough to modify for my exact needs in any scenario without a huge learning curve. I tried to leave everything accessable enough for you to do the same.
 
 ## This project has 3 main goals in mind.

 1. Do what I personally need for my projects (I hope it serves you too!)
 2. No complicated installation steps
 3. Something ready to use, fine-tuned right out of the box

 ## Startup:
 (Note: Some of this is temporary until I decide on a proper way of handling settings.)
 
 run.bat is included but you can feel free to use your environment of choice.
 
 ### The only thing you need to do is edit settings.py to point to your model paths. 

 ### The following startup flags are available:
```
 --all .................. Run everything (recommended)
 --api, --webui ......... Launch the FastAPI api and/or gradio webui with the following flags
 --tts, --llm, --sd ..... Enable text-to-speech, exllama2, and/or stable diffusion
```

### The following API endpoints are availble:
Text-to-speech:
```
/api/tts?model=<xtts|edge-tts>?text=<text>&voice=<voice>
```
Chat/text completion (OpenAI compatible):
```
see OpenAI documentation
```
Stable diffusion:
```
/api/sd?prompt=<prompt>&negative_prompt=<nehative_prompt>&guidance_scale=<guidance_scale>&steps=<steps>
```

## What is included/not included?
The only thing not included is EXL2 and Stable Diffusion models.
You will need to point settings.py to your paths.
The default page is a chat assistant that supports image generation, text/PDF import, chat history (stored locally) and more.
