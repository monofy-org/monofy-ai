import torch
from diffusers import StableVideoDiffusionPipeline
from settings import SD_MODEL, SD_USE_MODEL_VAE, SD_USE_SDXL
from utils.torch_utils import autodetect_device
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)

device = autodetect_device()


class SDClient:
    _instance = None

    @classmethod
    @property
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()  # Create an instance if it doesn't exist

        return cls._instance

    def __init__(self):
        self.generator = torch.manual_seed(42)
        self.image_pipeline = None
        self.video_pipeline = None

        self.video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="models/img2vid",
        )

        image_pipeline_type = (
            StableDiffusionXLPipeline if SD_USE_SDXL else StableDiffusionPipeline
        )

        self.image_pipeline = image_pipeline_type.from_single_file(
            SD_MODEL,
            variant="fp16",
            load_safety_checker=False,
            torch_dtype=torch.float16,
        )
        self.image_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.image_pipeline.scheduler.config
        )

        self.txt2img = AutoPipelineForText2Image.from_pipe(self.image_pipeline)
        self.img2img = AutoPipelineForImage2Image.from_pipe(self.image_pipeline)

        # if SD_USE_MODEL_VAE:
        #    vae = AutoencoderKL.from_single_file(
        #        SD_MODEL, variant="fp16", torch_dtype=torch.float16
        #    ).to(device)
        #    self.image_pipeline.vae = vae
