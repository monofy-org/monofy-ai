from typing import Generator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch
import os
from settings import LLM_DEFAULT_SEED, LLM_MODEL, USE_FP16
from huggingface_hub import snapshot_download

MODEL = "microsoft/phi-2"


class TransformersClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist
            cls._instance.load_model()
        return cls._instance

    def __init__(self):
        self.model_name = MODEL
        self.model = None
        self.tokenizer = None
        self.text_pipeline = None

        path = "models/llm/models--" + MODEL.replace("/", "--")
        if os.path.isdir(path):
            self.model_path = os.path.abspath(path)
        else:
            self.model_path = snapshot_download(
                repo_id=LLM_MODEL, cache_dir="models/llm", local_dir=path
            )

    def load_model(self, model_name=MODEL):
        if (self.model is None) or (model_name != self.model_name):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if USE_FP16 else torch.float32,
                variant="fp16" if USE_FP16 else None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=os.path.join("models", "llm"))
            self.text_pipeline = pipeline(self.model, torch_dtype=torch.float16)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_p=0.9,
        token_repetition_penalty=1.15,
        seed=LLM_DEFAULT_SEED,
    ) -> Generator[str, None, None]:
        self.load_model(self.model_name)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", return_attention_mask=False
        )
        outputs = self.model.generate(
            **inputs,
            max_length=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            token_repetition_penalty=token_repetition_penalty,
            seed=seed
        )
        text = self.tokenizer.batch_decode(outputs)[0]
        return text
