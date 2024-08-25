import logging
from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from classes.requests import Txt2VidRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.stable_diffusion import StableDiffusionPlugin
from plugins.video_plugin import VideoPlugin
from utils.gpu_utils import set_seed


class Txt2VidAnimatePlugin(VideoPlugin):
    name = "Animate"
    description = "Animate text-to-video generation"
    instance = None
    plugins = [StableDiffusionPlugin]

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

        self.resources["scheduler"] = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            original_inference_steps=100,
            steps_offset=1,
            timestep_scaling=10,  # default is 10
        )

        self.resources["adapter"] = motion_adapter

        self.current_lora = None

    async def load_model(self, model_index: int):
        from diffusers import AnimateDiffPipeline

        if self.resources.get("pipeline"):
            return self.resources["pipeline"]
        else:
            sd: StableDiffusionPlugin = await use_plugin(StableDiffusionPlugin, True)
            sd.load_model(model_index)

            pipe: AnimateDiffPipeline = AnimateDiffPipeline.from_pipe(
                sd.resources["pipeline"],
                motion_adapter=self.resources["adapter"],
                image_encoder=self.resources["image_encoder"],
                scheduler=self.resources["scheduler"],
            )
            pipe.load_lora_weights(
                "wangfuyun/AnimateLCM",
                weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                adapter_name="lcm-lora",
            )
            pipe.set_adapters(["lcm-lora"], [0.5])
            pipe.fuse_lora(adapter_names=["lcm-lora"])

            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

            self.resources["pipeline"] = pipe
            return pipe

    async def generate(
        self,
        req: Txt2VidRequest,
    ):

        if not req.num_inference_steps:
            req.num_inference_steps = 16

        pipe = await self.load_model(req.model_index)

        pipe.scheduler = self.resources["scheduler"]

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
