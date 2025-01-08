import asyncio
import logging
import huggingface_hub
import safetensors.torch
import tqdm.rich
from typing import Literal
from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
import torch
from classes.requests import Txt2ImgRequest, Txt2VidRequest
from plugins.video_plugin import VideoPlugin
from modules.filter import filter_prompt
from settings import SD_MODELS, USE_ACCELERATE, USE_XFORMERS
from utils.console_logging import log_generate, log_highpower, log_loading, log_recycle
from utils.gpu_utils import autodetect_device, clear_gpu_cache, set_seed
from modules.plugins import (
    PluginBase,
    check_low_vram,
    release_plugin,
    use_plugin,
)
from diffusers import (
    LCMScheduler,
    EulerAncestralDiscreteScheduler,
    AnimateDiffPipeline,
    TCDScheduler,
    DPMSolverMultistepScheduler,
    MotionAdapter,
)
from utils.stable_diffusion_utils import (
    load_lora_settings,
    load_prompt_lora,
    manual_offload,
)

lock = asyncio.Lock()


CLIP_MODELS: list[str] = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
]

MOTION_ADAPTERS: dict[str, str] = {
    "animatediff": "guoyww/animatediff-motion-adapter-v1-5-3",
    "animatelcm": "wangfuyun/AnimateLCM",
}

default_clip_models: dict[str, int] = {
    "animatediff": 0,
    "animatelcm": 0,
}


default_schedulers: dict[str, Literal["euler_a", "lcm", "sde", "tcd"]] = {
    "animatediff": "lcm",
    "animatelcm": "lcm",
}


class Txt2VidAnimatePlugin(VideoPlugin):
    name = "Text-to-video (AnimateDiff+AnimateLCM)"
    description = "Text-to-video generation using AnimateDiff and AnimateLCM"
    instance = None
    plugins = ["MMAudioPlugin"]

    def __init__(self):
        super().__init__()
        self.device = autodetect_device()
        self.current_motion_adapter: Literal["animatediff", "animatelcm"] = None
        self.current_model_index = -1
        self.current_clip_index = -1
        self.current_scheduler_type: Literal["euler_a", "lcm", "sde", "tcd"] = None
        self.current_weights_path = None
        self.current_lightning_steps = None
        self.modified_unet = False
        self.last_loras = []
        self.use_hidiffusion = False
        self.use_animatelcm = False

    def offload(self):
        manual_offload(self.resources.get("pipeline"))
        manual_offload(self.resources.get("sd"))

    def check_model_integrity(self, req: Txt2VidRequest):
        if not self.resources.get("pipeline"):
            return

        polluted = (
            (
                req.use_lightning
                and self.current_lightning_steps != req.num_inference_steps
            )
            or (req.use_lightning and not self.current_lightning_steps)
            or (self.use_animatelcm and not req.use_animatelcm)
            or (self.current_lightning_steps and not req.use_lightning)
            or (self.current_motion_adapter != req.motion_adapter)
        )

        if polluted or req.model_index != self.current_model_index:
            logging.info("Unloading previous pipeline...")
            del self.resources["pipeline"]
            del self.resources["sd"]
            self.use_animatelcm = False
            self.use_hidiffusion = False
            self.current_lightning_steps = None
            if polluted and self.resources.get("image_encoder"):
                del self.resources["image_encoder"]
                self.current_clip_index = -1
            check_low_vram()

    def get_image_encoder(self, clip_model_index: int):
        from transformers import CLIPVisionModelWithProjection

        image_encoder: CLIPVisionModelWithProjection = self.resources.get(
            "image_encoder"
        )

        if image_encoder and clip_model_index == self.current_clip_index:
            log_recycle(f"Reusing image encoder: {CLIP_MODELS[clip_model_index]}")
            return image_encoder

        if image_encoder:
            del self.resources["image_encoder"]

        log_loading("CLIP model", CLIP_MODELS[clip_model_index or 0])

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_MODELS[clip_model_index or 0]
        )

        self.current_clip_index = clip_model_index

        return image_encoder

    def get_motion_adapter(self, adapter_name: Literal["animatediff", "animatelcm"]):
        motion_adapter = self.resources.get("motion_adapter")
        if motion_adapter and adapter_name == self.current_motion_adapter:
            log_recycle(f"Reusing motion adapter {adapter_name}")
            return motion_adapter

        if motion_adapter:
            logging.info("Unloading previous motion adapter...")
            del self.resources["motion_adapter"]
            clear_gpu_cache()

        if adapter_name == "animatediff":
            log_loading("Motion adapter", "guoyww/animatediff-motion-adapter-v1-5-3")
            motion_adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-3",
                # torch_dtype=torch.float16,
                variant="fp16",
            )
        elif adapter_name == "animatelcm":
            log_loading("Motion adapter", "wangfuyun/AnimateLCM")
            motion_adapter = MotionAdapter.from_pretrained(
                "wangfuyun/AnimateLCM",
                # torch_dtype=torch.float16,
                variant="fp16",
            )
        else:
            raise ValueError(f"Invalid motion adapter: {adapter_name}")

        self.current_motion_adapter = adapter_name

        self.resources["motion_adapter"] = motion_adapter

        return motion_adapter

    def set_scheduler(
        self, scheduler_name: Literal["euler_a", "lcm", "sde", "tcd"], config=None
    ):
        scheduler: LCMScheduler | EulerAncestralDiscreteScheduler = self.resources.get(
            "scheduler"
        )
        if scheduler and scheduler_name == self.current_scheduler_type:
            log_recycle(f"Reusing scheduler: {scheduler_name}")
            return scheduler

        if scheduler_name == "euler_a":
            scheduler = (
                EulerAncestralDiscreteScheduler(
                    beta_start=0.00085,  # copied from lcm scheduler
                    beta_end=0.012,  # copied from lcm scheduler
                    steps_offset=1,  # copied from lcm scheduler
                )
                if config is None
                else EulerAncestralDiscreteScheduler.from_config(config)
            )
        elif scheduler_name == "tcd":
            scheduler = (
                TCDScheduler(
                    beta_start=0.00085,  # copied from lcm scheduler
                    beta_end=0.012,  # copied from lcm scheduler
                    steps_offset=1,  # copied from lcm scheduler
                )
                if config is None
                else TCDScheduler.from_config(config)
            )
        elif scheduler_name == "sde":
            scheduler = (
                DPMSolverMultistepScheduler(
                    beta_start=0.00085,  # copied from lcm scheduler
                    beta_end=0.012,  # copied from lcm scheduler
                    steps_offset=1,  # copied from lcm scheduler
                )
                if config is None
                else DPMSolverMultistepScheduler.from_config(config)
            )
        elif scheduler_name == "lcm" or not scheduler_name:
            scheduler = (
                LCMScheduler(
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="linear",
                    original_inference_steps=100,
                    steps_offset=1,
                    timestep_scaling=10,  # default is 10
                )
                if config is None
                else LCMScheduler.from_config(config)
            )
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_name}")

        self.current_scheduler_type = scheduler_name
        self.resources["scheduler"] = scheduler

        return scheduler

    def load_additional_unet_weights(self, weights_path: str):
        pipe: AnimateDiffPipeline = self.resources.get("pipeline")
        weights = safetensors.torch.load_file(weights_path, device=self.device)
        pipe.unet.load_state_dict(
            weights,
            strict=False,
        )
        del weights
        torch.cuda.empty_cache()

    async def load_model(self, req: Txt2VidRequest):
        if req.model_index < 0 or req.model_index >= len(SD_MODELS):
            raise ValueError(f"Invalid model index: {req.model_index}")

        if (
            req.use_lightning
            and req.num_inference_steps
            and req.num_inference_steps not in [1, 2, 4, 8]
        ):
            req.num_inference_steps = 8
            logging.warning(
                f"Lightning requires num_inference_steps to be 1, 2, 4, or 8. Defaulting to {req.num_inference_steps}."
            )
        elif not req.num_inference_steps:
            req.num_inference_steps = 16

        model_name = SD_MODELS[req.model_index]
        log_loading("model", model_name)

        image_encoder = self.get_image_encoder(
            req.clip_index
            if req.clip_index
            else default_clip_models[req.motion_adapter]
        )
        assert image_encoder is not None
        self.resources["image_encoder"] = image_encoder

        pipe = self.resources.get("pipeline", None)

        if (
            pipe
            and self.resources.get("sd", None)
            and self.resources.get("image_encoder", None)
            and self.resources.get("motion_adapter", None)
            and self.current_motion_adapter == req.motion_adapter
        ):
            pipe.image_encoder = self.resources["image_encoder"]
            return pipe

        motion_adapter = self.get_motion_adapter(req.motion_adapter)
        assert motion_adapter is not None
        self.resources["motion_adapter"] = motion_adapter

        from diffusers import StableDiffusionPipeline

        from_model = (
            StableDiffusionPipeline.from_single_file
            if model_name.endswith(".safetensors")
            else StableDiffusionPipeline.from_pretrained
        )

        sd_pipeline: StableDiffusionPipeline = from_model(
            model_name,
            torch_dtype=self.dtype,
            device=self.device,
            requires_safety_checker=False,
            image_encoder=image_encoder,
        )
        sd_pipeline.progress_bar = tqdm.rich.tqdm

        self.resources["sd"] = sd_pipeline
        self.current_model_index = req.model_index
        self.last_loras = []

        # create AnimateDiffPipeline
        pipe: AnimateDiffPipeline = AnimateDiffPipeline.from_pipe(
            sd_pipeline,
            image_encoder=image_encoder,
            motion_adapter=motion_adapter,
            scheduler=self.set_scheduler(req.scheduler),
        )

        pipe.enable_model_cpu_offload(None, self.device)

        pipe.progress_bar = tqdm.rich.tqdm

        if USE_XFORMERS and not USE_ACCELERATE:
            pipe.vae.enable_xformers_memory_efficient_attention()

        self.current_model_index = req.model_index
        self.resources["pipeline"] = pipe

        pipe.enable_vae_slicing()

        return pipe

    def load_animatelcm_lora(self, pipe: AnimateDiffPipeline):
        if not self.use_animatelcm:
            log_highpower(
                "Fusing AnimateLCM LoRA: wangfuyun/AnimateLCM/AnimateLCM_sd15_t2v_lora.safetensors",
            )

            pipe.unload_lora_weights()
            self.last_loras = []

            pipe.load_lora_weights(
                "wangfuyun/AnimateLCM",
                weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                adapter_name="lcm-lora",
                device=self.device,
            )

            pipe.set_adapters(
                ["lcm-lora"],
                adapter_weights=[0.8],
            )

            pipe.fuse_lora(["unet", "text_encoder"], 1.0, adapter_names=["lcm-lora"])
            pipe.unload_lora_weights()  # unload again after fusing
            self.use_animatelcm = True

    def load_lightning_weights(
        self,
        pipe: AnimateDiffPipeline,
        num_inference_steps: int,
    ):
        log_loading(
            "lightning weights",
            f"animatediff_lightning_{num_inference_steps}step_diffusers.safetensors",
        )
        weights_path = huggingface_hub.hf_hub_download(
            "ByteDance/AnimateDiff-Lightning",
            f"animatediff_lightning_{num_inference_steps}step_diffusers.safetensors",
        )
        pipe.load_lora_weights(
            weights_path,
            adapter_name="lcm-lora",
        )
        pipe.set_adapters(
            ["lcm-lora"],
            adapter_weights=[0.8],
        )

        pipe.fuse_lora(["unet", "text_encoder"], 1.0, adapter_names=["lcm-lora"])
        pipe.unload_lora_weights()  # unload again after fusing
        self.current_lightning_steps = num_inference_steps

    def set_hidiffusion(self, sd_pipeline, enable_hidiffusion: bool):
        if not sd_pipeline:
            raise ValueError("SD pipeline not loaded")

        if enable_hidiffusion:
            if not self.use_hidiffusion:
                log_highpower("Applying HiDiffusion weights")
                from submodules.HiDiffusion.hidiffusion.hidiffusion import (
                    apply_hidiffusion,
                )

                apply_hidiffusion(sd_pipeline)
        else:
            if self.use_hidiffusion:
                log_highpower("Removing HiDiffusion weights")
                from submodules.HiDiffusion.hidiffusion.hidiffusion import (
                    remove_hidiffusion,
                )

                remove_hidiffusion(sd_pipeline)

        if self.use_hidiffusion != enable_hidiffusion:
            pipe: AnimateDiffPipeline = self.resources["pipeline"]
            pipe.unet = sd_pipeline.unet
            self.use_hidiffusion = enable_hidiffusion
            pipe.enable_model_cpu_offload(None, self.device)

    async def generate(
        self,
        req: Txt2VidRequest,
    ):
        filter_prompt(req.prompt, req.negative_prompt, req.nsfw)

        self.check_model_integrity(req)

        from diffusers import AnimateDiffPipeline

        pipe: AnimateDiffPipeline = await self.load_model(req)

        if self.current_motion_adapter != req.motion_adapter:
            pipe.motion_adapter = self.get_motion_adapter(req.motion_adapter)

        if req.use_animatelcm:
            self.load_animatelcm_lora(pipe)

        if req.use_lightning:
            self.load_lightning_weights(pipe, req.num_inference_steps)

        self.set_hidiffusion(self.resources["sd"], req.hi)

        if req.scheduler != self.current_scheduler_type:
            self.current_scheduler = req.scheduler
            pipe.scheduler = self.set_scheduler(req.scheduler)

        logging.info(
            f"AnimateDiff (model={self.current_model_index}, motion_adapter={self.current_motion_adapter}, scheduler={req.scheduler}, hi={req.hi}, use_animatelcm={req.use_animatelcm}, use_lightning={self.current_lightning_steps}, clip={CLIP_MODELS[self.current_clip_index]}"
        )

        default_negs = "(low quality:1.5), disfigured, (blurry:1.5), flicker, noise, lens flare, disappearing, surreal, splotches, background movement, hands moving, watermark"

        req.negative_prompt = (
            default_negs
            if not req.negative_prompt
            else req.negative_prompt + ", " + default_negs
        )

        if req.auto_lora and req.lora_strength > 0:
            loras = load_prompt_lora(
                pipe,
                Txt2ImgRequest(**req.__dict__),
                load_lora_settings("sd15"),
                self.last_loras,
            )
            self.last_loras = loras

        _, generator = set_seed(req.seed, return_generator=True)

        log_generate(f"Generating video ({req.width}x{req.height})")

        output = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            width=req.width,
            height=req.height,
            num_frames=req.num_frames,
            generator=generator,
            do_classifier_free_guidance=True,
        )

        frames = output.frames[0]

        # req = Txt2ImgRequest(
        #     prompt=req.prompt,
        #     face_prompt=req.prompt,
        #     negative_prompt=req.negative_prompt,
        #     width=req.width,
        #     height=req.height,
        #     guidance_scale=req.guidance_scale,
        #     num_inference_steps=req.num_inference_steps,
        #     seed=req.seed,
        # )

        # frames = [inpaint_faces(inpaint, frame, req, False) for frame in frames]

        return frames


@PluginBase.router.post(
    "/txt2vid/animate",
    response_class=FileResponse,
    tags=["Video Generation (text-to-video)"],
)
async def txt2vid(
    background_tasks: BackgroundTasks,
    req: Txt2VidRequest,
):
    async with lock:
        plugin = None
        try:
            plugin: Txt2VidAnimatePlugin = await use_plugin(Txt2VidAnimatePlugin)
            frames = await plugin.generate(req)
            return plugin.video_response(
                background_tasks,
                frames,
                req,
            )

        except Exception as e:
            logging.exception(e)
            if isinstance(e, HTTPException):
                raise e
            else:
                raise HTTPException(status_code=500, detail="Internal error")
        finally:
            if plugin is not None:
                release_plugin(plugin)


@PluginBase.router.get(
    "/txt2vid/animate",
    response_class=FileResponse,
    tags=["Video Generation (text-to-video)"],
)
async def txt2vid_get(
    background_tasks: BackgroundTasks,
    req: Txt2VidRequest = Depends(),
):
    return await txt2vid(background_tasks, req)
