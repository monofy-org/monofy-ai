import logging
import safetensors.torch
from typing import Literal
from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
import torch
from classes.requests import Txt2ImgRequest, Txt2VidRequest
from plugins.video_plugin import VideoPlugin
from plugins.stable_diffusion import StableDiffusionPlugin
from modules.filter import filter_prompt
from settings import USE_ACCELERATE, USE_XFORMERS
from utils.console_logging import log_highpower, log_loading, log_recycle
from utils.gpu_utils import autodetect_device, clear_gpu_cache, set_seed
from modules.plugins import (
    PluginBase,
    release_plugin,
    unload_plugin,
    use_plugin,
    use_plugin_unsafe,
)
from diffusers import (
    LCMScheduler,
    EulerAncestralDiscreteScheduler,
    AnimateDiffPipeline,
    MotionAdapter,
)
from utils.stable_diffusion_utils import load_lora_settings, load_prompt_lora


CLIP_MODELS: list[str] = [
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch32",
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

default_schedulers: dict[str, Literal["euler_a", "lcm"]] = {
    "animatediff": "euler_a",
    "animatelcm": "lcm",
}


class Txt2VidAnimatePlugin(VideoPlugin):
    name = "Text-to-video (AnimateDiff+AnimateLCM)"
    description = "Text-to-video generation using AnimateDiff and AnimateLCM"
    instance = None
    plugins = ["StableDiffusionPlugin"]

    def __init__(self):

        super().__init__()
        self.device = autodetect_device()
        self.current_motion_adapter: Literal["animatediff", "animatelcm"] = None
        self.current_model_index = -1
        self.current_clip_index = -1
        self.current_scheduler_type: Literal["euler_a", "lcm"] = None
        self.current_weights_path = None
        self.current_lightning_steps = None
        self.modified_unet = False
        self.last_loras = []

        # self.lightning_weights_path = huggingface_hub.hf_hub_download(
        #     "ByteDance/AnimateDiff-Lightning",
        #     "animatediff_lightning_8step_diffusers.safetensors",
        # )

    def check_model_integrity(self, req: Txt2VidRequest):
        if not self.resources.get("pipeline"):
            return

        if (
            (req.model_index != self.current_model_index)
            or (req.use_lightning and not self.current_lightning_steps)
            or (self.current_lightning_steps and not req.use_lightning)
            or (
                req.use_lightning
                and self.current_lightning_steps != req.num_inference_steps
            )
        ):
            logging.info("Unloading previous pipeline...")
            # TODO: just reload the unet
            del self.resources["pipeline"]

    def set_image_encoder(self, clip_model_index: int):
        from transformers import CLIPVisionModelWithProjection

        image_encoder: CLIPVisionModelWithProjection = self.resources.get(
            "image_encoder"
        )

        if (
            image_encoder
            and self.current_clip_index > -1
            and clip_model_index == self.current_clip_index
        ):
            log_recycle(f"Reusing image encoder: {CLIP_MODELS[clip_model_index]}")
            return image_encoder

        if image_encoder:
            del self.resources["image_encoder"]

        log_loading("CLIP model", CLIP_MODELS[clip_model_index])

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_MODELS[clip_model_index]
        )

        self.current_clip_index = clip_model_index
        self.resources["image_encoder"] = image_encoder

        return image_encoder

    async def set_motion_adapter(
        self, adapter_name: Literal["animatediff", "animatelcm"]
    ):
        motion_adapter = self.resources.get("motion_adapter")
        if motion_adapter and adapter_name == self.current_motion_adapter:
            log_recycle(f"Reusing motion adapter {adapter_name}")
            return motion_adapter

        if motion_adapter:
            logging.info("Unloading previous motion adapter...")
            del self.resources["motion_adapter"]
            clear_gpu_cache()

        if adapter_name == "animatediff":
            motion_adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-3",
                # torch_dtype=torch.float16,
                variant="fp16",
            )
        elif adapter_name == "animatelcm":
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

    def set_scheduler(self, scheduler_name: Literal["euler_a", "lcm"], config=None):

        scheduler: LCMScheduler | EulerAncestralDiscreteScheduler = self.resources.get(
            "scheduler"
        )
        if scheduler and scheduler_name == self.current_scheduler_type:
            log_recycle(f"Reusing scheduler: {scheduler_name}")
            return scheduler

        if scheduler_name == "euler_a":
            scheduler = (
                EulerAncestralDiscreteScheduler(
                    # beta_start=0.00085,
                    # beta_end=0.012,
                    # beta_schedule="linear",
                    # num_train_timesteps=100,
                    steps_offset=1,
                    # timestep_scaling=10,  # default is 10
                )
                if config is None
                else EulerAncestralDiscreteScheduler.from_config(config)
            )
        elif scheduler_name == "lcm":
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

        if req.model_index < 0:
            raise ValueError("Invalid model index")

        self.check_model_integrity(req)

        # re-use existing model if possible
        pipe = self.resources.get("pipeline", None)
        sd: StableDiffusionPlugin = self.resources.get("StableDiffusionPlugin", None)
        sd_pipeline = sd.resources.get("pipeline") if sd else None

        # "Passively" below means no pipeline properties are changed, only self.resources is updated.
        # If no changes are necessary, these are no-ops.

        # passively update self.resources["image_encoder"] and return image_encoder
        image_encoder = self.set_image_encoder(
            req.clip_index
            if req.clip_index
            else default_clip_models[req.motion_adapter]
        )
        assert self.resources["image_encoder"] is not None

        # passively update self.resources["scheduler"] and return scheduler
        scheduler = self.set_scheduler(
            req.scheduler if req.scheduler else default_schedulers[req.motion_adapter]
        )
        assert self.resources["scheduler"] is not None

        # passively update self.resources["motion_adapter"] and return motion_adapter (async)
        motion_adapter = await self.set_motion_adapter(req.motion_adapter)
        assert self.resources["motion_adapter"] is not None

        if (
            pipe
            and sd_pipeline
            and req.model_index > -1
            and req.model_index == self.current_model_index
        ):
            # all conditions met, update and return existing pipeline
            pipe.image_encoder = image_encoder
            pipe.scheduler = scheduler
            pipe.motion_adapter = motion_adapter
            return pipe

        if sd and req.model_index != self.current_model_index:
            # unload previously-used plugin
            unload_plugin(StableDiffusionPlugin)

        # create new plugin instance
        sd = use_plugin_unsafe(StableDiffusionPlugin)
        self.resources["StableDiffusionPlugin"] = sd

        # load model
        # TODO: add a "force_reload" flag instead of doing the above stuff
        sd.load_model(req.model_index, image_encoder=image_encoder)

        # update properties / clear saved properties
        self.current_model_index = req.model_index
        self.last_loras = []
        self.resources["pipeline"] = None

        # get the currently-loaded StableDiffusionPipeline
        sd_pipeline = sd.resources.get("pipeline")
        assert sd_pipeline is not None

        sd_pipeline.image_encoder = image_encoder

        # create AnimateDiffPipeline
        pipe: AnimateDiffPipeline = AnimateDiffPipeline.from_pipe(
            sd_pipeline,
            image_encoder=image_encoder,
            motion_adapter=motion_adapter,
            scheduler=scheduler,
        ).to(self.device)

        if USE_XFORMERS and not USE_ACCELERATE:
            pipe.vae.enable_xformers_memory_efficient_attention()

        if "lcm-lora" not in pipe.get_active_adapters():

            log_loading(
                "AnimateLCM LoRA",
                "wangfuyun/AnimateLCM/AnimateLCM_sd15_t2v_lora.safetensors",
            )
            pipe.unload_lora_weights()

            pipe.load_lora_weights(
                "wangfuyun/AnimateLCM",
                weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                adapter_name="lcm-lora",
                device=self.device,
            )

            pipe.fuse_lora(["unet", "text_encoder"], 0.8, adapter_names=["lcm-lora"])
            pipe.unload_lora_weights()  # unload again after fusing

        self.current_model_index = req.model_index
        self.resources["pipeline"] = pipe

        # refresh favorites.json whether we need to or not
        self.resources["lora_settings"] = load_lora_settings("sd15")

        pipe.enable_vae_slicing()

        pipe.enable_model_cpu_offload(None, self.device)

        return pipe

    async def generate(
        self,
        req: Txt2VidRequest,
    ):
        filter_prompt(req.prompt, req.negative_prompt, req.nsfw)

        from diffusers import AnimateDiffPipeline

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

        pipe: AnimateDiffPipeline = await self.load_model(req)

        logging.info(
            f"AnimateDiff using model index {self.current_model_index} with {self.current_motion_adapter} and {CLIP_MODELS[self.current_clip_index]} (use_lightning={self.current_lightning_steps})"
        )

        default_negs = "(low quality:1.5), disfigured, (blurry:1.5), flicker, noise, lens flare, disappearing, surreal, splotches, background movement, hands moving, watermark"

        req.negative_prompt = (
            default_negs
            if not req.negative_prompt
            else req.negative_prompt + ", " + default_negs
        )

        loras = load_prompt_lora(
            pipe,
            Txt2ImgRequest(**req.__dict__),
            self.resources["lora_settings"],
            self.last_loras,
        )

        self.last_loras = loras

        _, generator = set_seed(req.seed, return_generator=True)

        pipe.image_encoder = self.resources["image_encoder"]
        pipe.scheduler = self.resources["scheduler"]

        log_highpower(f"Generating video ({req.width}x{req.height})")

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

        # inpaint = StableDiffusionPlugin.instance.resources["inpaint"]

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
    plugin = None
    try:
        plugin: Txt2VidAnimatePlugin = await use_plugin(Txt2VidAnimatePlugin)
        frames = await plugin.generate(req)
        return plugin.video_response(
            background_tasks,
            frames,
            req.fps,
            req.interpolate_film,
            req.interpolate_rife,
            req.fast_interpolate,
            req.audio,
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
