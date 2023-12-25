from diffusers import ShapEPipeline
from diffusers.utils import export_to_gif, export_to_ply
import torch
from settings import DEVICE, USE_FP16, USE_XFORMERS
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
            variant="fp16" if USE_FP16 else None,
            torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        )
        self.pipe.to(memory_format=torch.channels_last)

        if USE_XFORMERS:
            self.pipe.enable_xformers_memory_efficient_attention()

        if (torch.cuda.is_available()):
            self.pipe.enable_model_cpu_offload()

    def generate(
        self,
        prompt: str,
        file_path: str,
        steps: int = 32,
        guidance_scale: float = 15.0,
        format: str = "gif",
    ):
        free_vram("shap-e")
        if format == "gif":
            images = self.pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                frame_size=256,
            ).images[0]
        elif format == "ply":
            images = self.pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                frame_size=256,
                output_type="mesh",
            ).images[0]
        else:
            return None

        print(f"Saving {len(images)} images to {file_path}")

        if format == "ply":
            export_to_ply(images, file_path)

        else:
            export_to_gif(images, file_path)

        return file_path
