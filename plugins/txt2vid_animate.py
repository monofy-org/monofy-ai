import logging
from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from classes.requests import Txt2ImgRequest, Txt2VidRequest
from modules.filter import filter_prompt
from modules.plugins import (
    PluginBase,
    release_plugin,
    unload_plugin,
    use_plugin,
    use_plugin_unsafe,
)
from plugins.stable_diffusion import StableDiffusionPlugin
from plugins.video_plugin import VideoPlugin
from utils.gpu_utils import set_seed
from diffusers import LCMScheduler

from utils.stable_diffusion_utils import load_lora_settings, load_prompt_lora

recommended_scheduler = LCMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="linear",
    original_inference_steps=100,
    steps_offset=1,
    timestep_scaling=10,  # default is 10
)


class Txt2VidAnimatePlugin(VideoPlugin):
    name = "Text-to-Video (AnimateDiff+AnimateLCM)"
    description = "Text-to-video generation using AnimateDiff and AnimateLCM"
    instance = None

    def __init__(self):
        import torch
        from transformers import CLIPVisionModelWithProjection
        from diffusers import LCMScheduler, MotionAdapter

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # motion_adapter = MotionAdapter.from_pretrained(
        #     "guoyww/animatediff-motion-adapter-sdxl-beta", variant="fp16", torch_dtype=torch.float16
        # )

        motion_adapter = MotionAdapter.from_pretrained(
            "wangfuyun/AnimateLCM", torch_dtype=torch.float16
        )

        self.resources["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
            # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )

        self.resources["scheduler"] = recommended_scheduler

        self.resources["adapter"] = motion_adapter

        self.last_loras = []

        self.current_model_index = -1

    async def load_model(self, model_index: int):
        from diffusers import AnimateDiffPipeline

        if model_index > -1 and model_index == self.current_model_index:
            return self.resources["pipeline"]

        sd: StableDiffusionPlugin = self.resources.get("StableDiffusionPlugin")

        if sd:
            unload_plugin(StableDiffusionPlugin)

        sd = use_plugin_unsafe(StableDiffusionPlugin)
        self.resources["StableDiffusionPlugin"] = sd

        sd.load_model(model_index)
        self.current_model_index = model_index
        self.last_loras = []
        self.resources["pipeline"] = None

        if self.resources.get("pipeline"):
            return self.resources["pipeline"]
        else:
            pipe: AnimateDiffPipeline = AnimateDiffPipeline.from_pipe(
                sd.resources["pipeline"],
                motion_adapter=self.resources["adapter"],
                image_encoder=self.resources["image_encoder"],
                scheduler=self.resources["scheduler"],
            )

            if "lcm-lora" not in pipe.get_active_adapters():
                pipe.unload_lora_weights()
                pipe.load_lora_weights(
                    "wangfuyun/AnimateLCM",
                    weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                    adapter_name="lcm-lora",
                )
                pipe.fuse_lora(["unet", "text_encoder"], 0.55)
                pipe.unload_lora_weights()  # unload again after fusing

            self.resources["lora_settings"] = load_lora_settings("sd15")

            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

            self.resources["pipeline"] = pipe

            self.current_model_index = model_index

            return pipe        


    async def generate(
        self,
        req: Txt2VidRequest,
    ):
        filter_prompt(req.prompt, req.negative_prompt, req.nsfw)

        from diffusers import AnimateDiffPipeline

        if not req.num_inference_steps:
            req.num_inference_steps = 16

        logging.info(f"AnimateDiff using model index {req.model_index}")

        pipe: AnimateDiffPipeline = await self.load_model(req.model_index)

        pipe.enable_model_cpu_offload()

        pipe.scheduler = self.resources["scheduler"]

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
            0.9,
        )

        # pipe.set_adapters(["lcm-lora"], [0.5])

        self.last_loras = loras

        set_seed(req.seed)

        args = req.__dict__

        output = pipe(**args, do_classifier_free_guidance=True)

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
