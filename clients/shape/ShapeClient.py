from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif
import torch
from settings import DEVICE, USE_XFORMERS
from utils.gpu_utils import free_vram, get_seed


class ShapeClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist

        return cls._instance

    def __init__(self):
        self.generator = get_seed(-1)
        self.pipe = ShapEPipeline.from_pretrained(
            "openai/shap-e",
            device=DEVICE,
            variant="fp16",
            torch_dtype=torch.float16,            
        )
        self.pipe.to(memory_format=torch.channels_last)

        if USE_XFORMERS:
            self.pipe.enable_xformers_memory_efficient_attention()

        self.pipe.enable_model_cpu_offload()
        free_vram(None)

    def generate(
        self, prompt: str, file_path: str, steps: int = 24, guidance_scale: float = 15.0
    ):
        free_vram("shape-e")
        images = self.pipe(
            prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            frame_size=192,
        ).images[0]

        print(f"Saving {len(images)} images to {file_path}")
        export_to_gif(images, file_path)

        return file_path
