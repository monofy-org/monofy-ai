import gc
import logging
import time
import torch
from utils.file_utils import import_model
from utils.misc_utils import print_completion_time


class ClientBase:

    _instance = None

    def __init__(self, friendly_name: str):
        self.models = []
        self.friendly_name = friendly_name
        self.current_model_name = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(
        self,
        model_type,
        model_name,
        unload_previous_model=True,
        allow_fp16=True,
        allow_bf16=True,
        set_variant_fp16: bool = None,
        **kwargs,
    ):
        if unload_previous_model:
            logging.info(f"Unloading previous model for {self.friendly_name}...")
            self.unload()

        self.current_model_name = model_name

        start_time = time.time()

        set_variant_fp16 = (
            set_variant_fp16 if set_variant_fp16 is not None else allow_fp16
        )

        model = import_model(
            model_type,
            model_name,
            allow_fp16=allow_fp16,
            allow_bf16=allow_bf16,
            set_variant_fp16=set_variant_fp16,
            **kwargs,
        )

        print_completion_time(start_time, "Model load")

        self.models.append(model)

    def offload(self, for_task: str):
        self.unload()

    def unload(self):
        for model in self.models:
            if hasattr(model, "unload"):
                model.unload()
            del model

        self.models.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
