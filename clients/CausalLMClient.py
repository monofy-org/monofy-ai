from typing import Generator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch
import os
from settings import LLM_DEFAULT_SEED, LLM_MODEL
from huggingface_hub import snapshot_download
from utils.gpu_utils import autodetect_device, autodetect_dtype
from utils.gpu_utils import use_fp16

MODEL = "microsoft/phi-2"


model_name = MODEL
model = None
tokenizer = None
text_pipeline = None

model_path = os.path.join("models", LLM_MODEL)

if not os.path.isdir(model_path):
    model_path = snapshot_download(
        repo_id=LLM_MODEL,
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )


def load_model(model_name=MODEL):
    global model_path
    global model
    global tokenizer
    global text_pipeline

    if (model is None) or (model_name != model_name):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device=autodetect_device(),
            torch_dtype=autodetect_dtype(),
            variant="fp16" if use_fp16 else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=os.path.join("models", "llm")
        )
        text_pipeline = pipeline(model, torch_dtype=torch.float16)


def generate_text(
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    top_p=0.9,
    token_repetition_penalty=1.15,
    seed=LLM_DEFAULT_SEED,
    frequency_penalty=1.15,
) -> Generator[str, None, None]:
    global model
    load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(
        **inputs,
        max_length=max_new_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        top_p=top_p,
        token_repetition_penalty=token_repetition_penalty,
        seed=seed
    )
    text = tokenizer.batch_decode(outputs)[0]
    return text
