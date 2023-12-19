import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer, load_quantized_model
from transformers.pipelines import pipeline
import torch
import os
from accelerate import init_empty_weights
from settings import LLM_MODEL, USE_FP16
from utils.gpu_utils import autodetect_device
from huggingface_hub import snapshot_download

MODEL = "TheBloke/dolphin-2.2.1-mistral-7B-GPTQ"


class GPTQClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist
            cls._instance.load_model()
        return cls._instance

    def __init__(self):
        self.tokenizer = None

        path = "models/llm/models--" + MODEL.replace("/", "--")
        if os.path.isdir(path):
            self.model_path = os.path.abspath(path)
        else:
            self.model_path = snapshot_download(
                repo_id=LLM_MODEL, cache_dir="models/llm", local_dir=path
            )

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if USE_FP16 else torch.float32,
            variant="fp16" if USE_FP16 else None,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.text_pipeline = pipeline(self.model, torch_dtype=torch.float16)
