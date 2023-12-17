import torch
from diffusers import StableVideoDiffusionPipeline
from settings import SD_MODEL, SD_USE_VAE, SD_USE_SDXL, USE_CUDAGRAPH, USE_XFORMERS
from utils.torch_utils import autodetect_device
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    ConsistencyDecoderVAE,
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
        self.generator = torch.manual_seed(-1)
        self.image_pipeline: AutoPipelineForText2Image = None
        self.video_pipeline: AutoPipelineForImage2Image = None
        self.inpainting_pipeline: AutoPipelineForInpainting = None

        self.video_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="models/img2vid",
        )

        image_pipeline_type = (
            StableDiffusionXLPipeline if SD_USE_SDXL else StableDiffusionPipeline
        )
        image_scheduler_type = (
            LMSDiscreteScheduler if SD_USE_SDXL else EulerDiscreteScheduler
        )

        self.image_pipeline = image_pipeline_type.from_single_file(
            SD_MODEL,
            variant="fp16",
            torch_dtype=torch.float16,
            enable_cuda_graph=USE_CUDAGRAPH,
        )

        self.image_pipeline.scheduler = image_scheduler_type.from_config(
            self.image_pipeline.scheduler.config
        )

        self.txt2img = AutoPipelineForText2Image.from_pipe(
            self.image_pipeline, safety_checker=None, requires_safety_checker=False
        )

        self.img2img = AutoPipelineForImage2Image.from_pipe(
            self.image_pipeline, safety_checker=None, requires_safety_checker=False
        )

        self.inpainting_pipeline = AutoPipelineForInpainting.from_pipe(
            self.image_pipeline, safety_checker=None, requires_safety_checker=False
        )

        self.image_pipeline.to(device)
        self.image_pipeline.enable_model_cpu_offload(0)

        self.video_pipeline.to(device)
        # self.video_pipeline.enable_model_cpu_offload(0)
        self.video_pipeline.enable_sequential_cpu_offload(0)

        if SD_USE_VAE:
            self.vae = ConsistencyDecoderVAE.from_config("openai/consistency-decoder")            
            self.image_pipeline.vae = self.vae

        if USE_XFORMERS:
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

            self.image_pipeline.enable_xformers_memory_efficient_attention(
                attention_op=MemoryEfficientAttentionFlashAttentionOp
            )
            self.image_pipeline.vae.enable_xformers_memory_efficient_attention(
                attention_op=None  # skip attention op for VAE
            )
