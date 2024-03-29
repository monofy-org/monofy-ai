import logging
from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from classes.requests import Txt2VidRequest
from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.gpu_utils import set_seed
from utils.video_utils import video_response


T2V_WEIGHTS = "AnimateLCM_sd15_t2v_lora.safetensors"


class Txt2VidAnimatePlugin(PluginBase):
    name = "Animate"
    description = "Animate text-to-video generation"
    instance = None

    def __init__(self):
        import torch
        from transformers import CLIPVisionModelWithProjection
        from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter

        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        motion_adapter = MotionAdapter.from_pretrained(
            "wangfuyun/AnimateLCM", torch_dtype=torch.float16
        )  # .to(device=self.device, dtype=self.dtype)

        clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            original_inference_steps=100,
            steps_offset=1,
            timestep_scaling=60, # default is 10
        )        

        pipe: AnimateDiffPipeline = AnimateDiffPipeline.from_pretrained(
            "emilianJR/epiCRealism",                
            motion_adapter=motion_adapter,
            torch_dtype=torch.float16,
            image_encoder=clip_vision_model,
            scheduler=scheduler,
        )  # .to(device=self.device, dtype=self.dtype)

        # state_dict = {}
        # with safetensors.safe_open(SD_MODELS[3], framework="pt", device="cpu") as f:
        #     for key in f.keys():
        #         state_dict[key] = f.get_tensor(key)
        # pipe.unet.load_state_dict(state_dict)
            

        pipe.load_lora_weights(
            "wangfuyun/AnimateLCM",
            weight_name=T2V_WEIGHTS,
            adapter_name="lcm-lora",
        )

        # pipe.enable_free_init()

        # pipe.scheduler = LCMScheduler.from_config(
        #     pipe.scheduler.config, beta_schedule="linear"
        # )
        self.resources["pipeline"] = pipe
        self.resources["scheduler"] = pipe.scheduler
        self.resources["adapter"] = motion_adapter
        self.resources["image_encoder"] = clip_vision_model

        # self.resources["lora_settings"] = load_lora_settings()

        self.current_lora = None

        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()

    async def generate(
        self,
        req: Txt2VidRequest,
    ):
        from diffusers import AnimateDiffPipeline

        pipe: AnimateDiffPipeline = self.resources["pipeline"]

        set_seed(req.seed)

        args = req.__dict__

        output = pipe(**args)

        frames = output.frames[0]

        # plugin: StableDiffusionPlugin = await use_plugin(StableDiffusionPlugin, True)
        # inpaint = plugin.resources["inpaint"]

        # req = Txt2ImgRequest(
        #     prompt=req.prompt,
        #     face_prompt=req.prompt,
        #     negative_prompt=req.negative_prompt,
        #     width=req.width,
        #     height=req.height,
        #     guidance_scale=req.guidance_scale,
        #     num_inference_steps=req.num_inference_steps,
        #     seed=seed,
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
        return video_response(background_tasks, frames, req)

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
