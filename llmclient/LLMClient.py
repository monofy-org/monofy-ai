from typing import Generator
import re
import logging
from TTS.utils.generic_utils import get_user_data_dir
from utils.torch_utils import autodetect_device
from utils.text_utils import process_text_for_llm
from huggingface_hub import snapshot_download

from settings import (
    LLM_MODEL,
    LOG_LEVEL,
    LLM_DEFAULT_SEED,
    LLM_GPU_SPLIT,
    LLM_MAX_SEQ_LEN,
    LLM_SCALE_POS_EMB,
    LLM_STOP_CONDITIONS,
    LLM_VALID_ENDINGS,
)
import time
import os
from typing import List
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler,
)


logging.basicConfig(level=LOG_LEVEL)


class LLMClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist
            cls._instance.load_model()
        return cls._instance

    def __init__(self):
        self.device = autodetect_device()
        logging.info(f"LLM using device: {self.device}")

        self.model_name = None
        self.model_path = None
        self.config = None
        self.model = None
        self.cache = None
        self.tokenizer = None
        self.generator = None
        self.streaming_generator = None
        self.user_name = "User"
        self.assistant_name = "Assistant"
        self.context = f"Considering the following conversation between {self.user_name} and {self.assistant_name}, give a single response as {self.assistant_name}. Do not prefix with your own name. Do not prefix with emojis."
        self.refresh_context()

    def load_model(self, model_name=LLM_MODEL):
        if model_name != self.model_name:
            path = "models/llm/models--" + LLM_MODEL.replace("/", "--")
            if os.path.isdir(path):
                self.model_path = os.path.abspath(path)
            else:            
              self.model_path = snapshot_download(
                  repo_id=LLM_MODEL, cache_dir="models/llm", local_dir=path
              )
              print(self.model_path)
            self.model_name = model_name
            self.config = ExLlamaV2Config()
            self.config.model_dir = self.model_path
            self.config.prepare()
            self.config.max_seq_len = LLM_MAX_SEQ_LEN
            self.config.scale_pos_emb = LLM_SCALE_POS_EMB
            # self.config.set_low_mem = True

            if self.model:
                logging.info("Unloading existing model...")
                self.model.unload()

            self.model = ExLlamaV2(self.config, lazy_load=True)
            logging.info("Loading model: " + model_name)

            self.cache = ExLlamaV2Cache(self.model, lazy=True)
            self.model.load_autosplit(self.cache, LLM_GPU_SPLIT)

            self.tokenizer = ExLlamaV2Tokenizer(self.config)
            self.generator = ExLlamaV2BaseGenerator(
                self.model, self.cache, self.tokenizer
            )

            self.streaming_generator = ExLlamaV2StreamingGenerator(
                self.model, self.cache, self.tokenizer
            )
            # self.streaming_generator.warmup()
            stop_conditions = [self.tokenizer.eos_token_id] + LLM_STOP_CONDITIONS

            self.streaming_generator.set_stop_conditions(stop_conditions)

        # Warm up regardless?
        else:
            self.generator.warmup()

    def refresh_context(self):
        try:
            with open("context.txt", "r") as file:
                self.context = file.read()
                logging.warn("Refreshed settings via API request.")
        except Exception:
            logging.error("Error reading context.txt, using default.")

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_p=0.9,
        chunk_sentences: bool = False,
        token_repetition_penalty=1.15,
        seed=LLM_DEFAULT_SEED,
    ) -> Generator[str, None, None]:
        self.load_model(self.model_name)

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = 20
        settings.top_p = top_p
        settings.token_repetition_penalty = 1.15
        settings.typical = 1.0
        settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        time_begin = time.time()

        print(f"\nFull text:\n---\n{prompt}\n---\n")

        generated_tokens = 0

        logging.info("Streaming response...")
        input_ids = self.tokenizer.encode(prompt)
        self.streaming_generator.begin_stream(input_ids, settings, True)

        message = ""
        sentence_count = 0
        i = 0

        while True:
            i += 1
            if i > LLM_MAX_SEQ_LEN:
                break  # *probably* impossible

            chunk, eos, _ = self.streaming_generator.stream()
            generated_tokens += 1

            # Never start a sentence with an emoji.
            # This quickly results in *every* sentence starting with an emoji.
            if message == "":
                chunk = process_text_for_llm(chunk)

            message += chunk

            if chunk_sentences:
                # Check if there's a complete sentence in the message
                if any(punctuation in message for punctuation in [".", "?", "!"]):
                    # Split the message into sentences and yield each sentence
                    sentences = re.split(r"(?<=[.!?])\s+", message)
                    for sentence in sentences[:-1]:
                        yield (
                            " " if sentence_count > 0 else ""
                        ) + process_text_for_llm(sentence)
                        sentence_count += 1
                    message = sentences[
                        -1
                    ]  # Keep the incomplete sentence for the next iteration

            else:
                yield process_text_for_llm(chunk)

            if eos or generated_tokens == max_new_tokens:
                break

        if (
            chunk_sentences
            and message
            and (message[-1] in LLM_VALID_ENDINGS)
            or message[-3:] == "```"
        ):
            yield (" " if sentence_count > 0 else "") + process_text_for_llm(message)

        time_end = time.time()
        time_total = time_end - time_begin

        print()
        print(
            f"Response generated in {time_total:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_total:.2f} tokens/second"
        )

    def chat(
        self,
        text: str,
        messages: List[dict],
        context="",
        max_new_tokens=80,
        temperature=0.7,
        top_p=0.9,
        chunk_sentences=True,
    ):
        prompt = (
            "System: " + (context + "\n\n" + self.context + "\n")
            if context
            else f"{self.context}\n"
        )

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            name = self.user_name if role == "user" else self.assistant_name
            prompt += f"\n\n{name}: {content}"

        if text is not None:
            prompt += f"\n\n{self.user_name}: {text}"

        prompt += f"\n\n{self.assistant_name}: "

        return self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            chunk_sentences=chunk_sentences,
        )
