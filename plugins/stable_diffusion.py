import os
import logging
import torch
from PIL import Image
from classes.requests import Txt2ImgRequest
from utils.gpu_utils import autodetect_dtype, clear_gpu_cache, set_seed
from typing import Literal
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import hf_hub_download
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.hypertile_utils import hypertile
from utils.image_utils import (
    crop_and_resize,
    get_image_from_request,
    image_to_base64_no_header,
    image_to_bytes,
)
from settings import (
    SD_DEFAULT_MODEL_INDEX,
    SD_HALF_VAE,
    SD_MODELS,
    SD_USE_HYPERTILE,
    SD_USE_LIGHTNING_WEIGHTS,
    SD_USE_SDXL,
    USE_XFORMERS,
    SD_COMPILE_UNET,
    SD_COMPILE_VAE,
    SD_USE_TOKEN_MERGING,
    SD_USE_DEEPCACHE,
)
from utils.stable_diffusion_utils import (
    enable_freeu,
    disable_freeu,
    filter_request,
    load_lora_settings,
    load_prompt_lora,
    postprocess,
)

YOLOS_MODEL = "hustvl/yolos-tiny"


class helpers:

    def get_model(repo_or_path: str):

        if os.path.exists(repo_or_path):
            return os.path.abspath(repo_or_path)

        elif repo_or_path.endswith(".safetensors"):

            # see if it is a valid repo/name/file.safetensors
            parts = repo_or_path.split("/")
            if len(parts) == 3:
                repo = parts[0]
                name = parts[1]
                file = parts[2]
                if not file.endswith(".safetensors"):
                    raise ValueError(
                        f"Invalid model path {repo_or_path}. Must be a valid local file or hf repo/name/file.safetensors"
                    )

                path = os.path.join("models", "Stable-diffusion")

                if os.path.exists(f"{path}/{file}"):
                    model_path = f"{path}/{file}"

                else:
                    repo_id = f"{repo}/{name}"
                    logging.info(f"Fetching {file} from {repo_id}...")
                    hf_hub_download(
                        repo_id,
                        filename=file,
                        local_dir=path,
                        local_dir_use_symlinks=False,
                        force_download=True,
                    )
                    model_path = os.path.join(path, file)

            else:
                raise FileNotFoundError(f"Model not found at {model_path}")

            return os.path.abspath(model_path)

        return repo_or_path


class StableDiffusionPlugin(PluginBase):

    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )
    from nudenet import NudeDetector

    name = "Stable Diffusion"
    description = "Base plugin for txt2img, img2img, inpaint, etc."
    instance = None

    def __init__(
        self,
        pipeline_type=(
            StableDiffusionXLPipeline if SD_USE_SDXL else StableDiffusionPipeline
        ),
        **model_kwargs,
    ):
        from diffusers import (
            SchedulerMixin,
        )
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        from nudenet import NudeDetector

        super().__init__()

        self.pipeline_type = pipeline_type
        self.dtype = autodetect_dtype(False)
        self.schedulers: dict[SchedulerMixin] = {}
        self.model_index = None
        self.model_kwargs = model_kwargs
        self.num_steps = 14

        self.resources["AutoImageProcessor"] = AutoImageProcessor.from_pretrained(
            YOLOS_MODEL,
        )

        self.resources["AutoModelForObjectDetection"] = (
            AutoModelForObjectDetection.from_pretrained(
                YOLOS_MODEL,
            )
        )
        self.resources["NudeDetector"] = NudeDetector()

        self.resources["lora_settings"] = load_lora_settings()

        # if SD_USE_DEEPCACHE:
        #     from DeepCache import DeepCacheSDHelper

        #     helper = DeepCacheSDHelper(pipe=image_pipeline)
        #     helper.set_params(
        #         cache_interval=3,
        #         cache_branch_id=0,
        #     )
        #     helper.enable()
        #     self.resources["DeepCacheSDHelper"] = helper

    def _load_model(self, model_index: int = SD_DEFAULT_MODEL_INDEX):

        if model_index == self.model_index:
            return

        self.model_index = model_index

        self.num_steps = (
            10
            if "lightning" in SD_MODELS[model_index].lower()
            else (
                14
                if "turbo" in SD_MODELS[model_index].lower()
                else 16 if SD_USE_SDXL else 25
            )
        )

        logging.info(f"Loading model index: {model_index}: {SD_MODELS[model_index]}")

        import torch
        from diffusers import (
            AutoencoderKL,
            AutoPipelineForText2Image,
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
        )

        repo_or_path = SD_MODELS[model_index]

        if self.resources.get("pipeline") is not None:
            del self.resources["txt2img"]
            del self.resources["img2img"]
            del self.resources["inpaint"]

            self.resources["pipeline"].unload_lora_weights()
            self.resources["pipeline"].maybe_free_model_hooks()                        
            del self.resources["pipeline"]

            clear_gpu_cache()

        model_path: str = helpers.get_model(repo_or_path)

        logging.info(f"Loading model: {os.path.basename(model_path)}")

        single_file = repo_or_path.endswith(".safetensors")

        from_model = (
            self.pipeline_type.from_single_file
            if single_file
            else self.pipeline_type.from_pretrained
        )

        kwargs = dict(device=self.device, torch_dtype=self.dtype)

        if "xl" in model_path.lower() and SD_HALF_VAE and self.dtype != torch.float32:
            kwargs["vae"] = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=self.dtype,
                use_safetensors=True,
            )

        image_pipeline = from_model(model_path, **kwargs, **self.model_kwargs).to(
            dtype=self.dtype
        )
        self.resources["pipeline"] = image_pipeline


        if "lightning" in model_path.lower() and SD_USE_LIGHTNING_WEIGHTS:
            logging.info("Loading SDXL Lightning LoRA weights...")
            image_pipeline.load_lora_weights(
                hf_hub_download(
                    "ByteDance/SDXL-Lightning", "sdxl_lightning_8step_lora.safetensors"
                )
            )
            image_pipeline.fuse_lora()

        # if "lightning" in model_path.lower():
        from diffusers import EulerDiscreteScheduler

        #     logging.info("Loading SDXL Lightning weights...")
        #     image_pipeline.unet.load_state_dict(
        #         load_file(
        #             hf_hub_download(
        #                 "ByteDance/SDXL-Lightning",
        #                 "sdxl_lightning_2step_unet.safetensors",
        #             ),
        #             device="cuda",
        #         )
        #     )

        image_pipeline.scheduler = EulerDiscreteScheduler.from_config(
            image_pipeline.scheduler.config, timestep_spacing="trailing"
        )

        # compile model (linux only)
        if not os.name == "nt":
            if SD_COMPILE_UNET:
                image_pipeline.unet = torch.compile(
                    image_pipeline.unet, mode="reduce-overhead", fullgraph=True
                )
            if SD_COMPILE_VAE:
                image_pipeline.vae = torch.compile(
                    image_pipeline.vae, mode="reduce-overhead", fullgraph=True
                )

        self.init_schedulers(image_pipeline)

        from diffusers.models.attention_processor import AttnProcessor2_0

        image_pipeline.unet.set_attn_processor(AttnProcessor2_0())

        # if SD_USE_TOKEN_MERGING:
        #     import tomesd
        #     tomesd.apply_patch(image_pipeline, ratio=0.5)

        if USE_XFORMERS:
            image_pipeline.enable_xformers_memory_efficient_attention()
            image_pipeline.vae.enable_xformers_memory_efficient_attention()

        image_pipeline.enable_model_cpu_offload()
        image_pipeline = image_pipeline.to(
            memory_format=torch.channels_last
        )

        if not SD_USE_HYPERTILE:
            image_pipeline.enable_vae_slicing()
            image_pipeline.enable_vae_tiling()

        self.default_scheduler = image_pipeline.scheduler

        image_pipeline.unet.to(memory_format=torch.channels_last)

        if self.__class__ == StableDiffusionPlugin:

            pipe_kwargs = dict(
                pipeline=image_pipeline, device=self.device, dtype=self.dtype
            )
            self.resources["txt2img"] = AutoPipelineForText2Image.from_pipe(
                **pipe_kwargs
            )
            self.resources["img2img"] = AutoPipelineForImage2Image.from_pipe(
                **pipe_kwargs
            )
            self.resources["inpaint"] = AutoPipelineForInpainting.from_pipe(
                **pipe_kwargs
            )

    def format_response(req: Txt2ImgRequest, response):

        if req.return_json:
            return JSONResponse(response)

        return StreamingResponse(
            image_to_bytes(response),
            media_type="image/png",
        )

    def init_schedulers(self, image_pipeline):
        from diffusers import (
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverSDEScheduler,
            LMSDiscreteScheduler,
            HeunDiscreteScheduler,
            DDIMScheduler,
            DDPMScheduler,
        )

        self.schedulers["euler"] = EulerDiscreteScheduler.from_config(
            image_pipeline.scheduler.config
        )
        self.schedulers["euler_a"] = EulerAncestralDiscreteScheduler.from_config(
            image_pipeline.scheduler.config
        )
        self.schedulers["sde"] = DPMSolverSDEScheduler.from_config(
            image_pipeline.scheduler.config
        )
        self.schedulers["lms"] = LMSDiscreteScheduler.from_config(
            image_pipeline.scheduler.config
        )
        self.schedulers["heun"] = HeunDiscreteScheduler.from_config(
            image_pipeline.scheduler.config
        )
        self.schedulers["ddim"] = DDIMScheduler.from_config(
            image_pipeline.scheduler.config
        )
        self.schedulers["ddpm"] = DDPMScheduler.from_config(
            image_pipeline.scheduler.config
        )

    async def generate(
        self,
        mode: Literal["txt2img", "img2img", "inpaint"],
        req: Txt2ImgRequest,
        **external_kwargs,
    ):
        req = filter_request(req)

        #print("generate_image", self.__class__.name)
        #print("prompt:", req.prompt)
        #print("negative_prompt:", req.negative_prompt)

        self._load_model(req.model_index)

        if not req.num_inference_steps:
            req.num_inference_steps = self.num_steps

        image_pipeline = self.resources["pipeline"]
        if req.freeu:
            enable_freeu(image_pipeline)
        else:
            disable_freeu(image_pipeline)

        if req.scheduler is not None:
            image_pipeline.scheduler = self.schedulers.get(
                req.scheduler, self.default_scheduler
            )

        image_pipeline.scheduler.config["lower_order_final"] = not SD_USE_SDXL
        image_pipeline.scheduler.config["use_karras_sigmas"] = True

        if req.auto_lora:
            lora_settings = self.resources["lora_settings"]
            load_prompt_lora(image_pipeline, req, lora_settings)

        pipe = self.resources[mode] if self.resources.get(mode) else image_pipeline

        req.seed = set_seed(req.seed)

        generator = torch.Generator(device="cuda")
        generator.manual_seed(req.seed)

        width = req.width // 128 * 128
        height = req.height // 128 * 128

        args = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            generator=generator,
        )

        if req.image is not None:
            args["image"] = [get_image_from_request(req.image)]
            logging.info(f"Using provided image as input for {mode}.")

        if SD_USE_HYPERTILE:
            result = hypertile(pipe, **args, **external_kwargs)
        else:
            result = pipe(**args, **external_kwargs)

        image = result.images[0]

        pipe.unload_lora_weights()

        if self.__class__ == StableDiffusionPlugin:
            image, json_response = await postprocess(self, image, req)
            if req.return_json:
                json_response["images"] = [image_to_base64_no_header(image)]
                return json_response
        else:
            if req.return_json:
                return {
                    "objects": [],
                    "detections": [],
                    "images": [image_to_base64_no_header(image)],
                }

        return image

    async def upscale_with_img2img(
        self,
        image: Image.Image,
        req: Txt2ImgRequest,
    ):
        if req.num_inference_steps > 100:
            logging.warn(f"Limiting steps to 100 from {req.num_inference_steps}")
            num_inference_steps = 100
        else:
            num_inference_steps = req.num_inference_steps

        if req.strength > 2:
            logging.warn(f"Limiting strength to 2 from {req.strength}")
            strength = 2
        else:
            strength = req.strength

        width = int((req.width * req.upscale) // 128 * 128)
        height = int((req.height * req.upscale) // 128 * 128)

        upscaled_image = crop_and_resize(image, (width, height))

        set_seed(req.seed)

        req: Txt2ImgRequest = req.copy()
        req.num_inference_steps = num_inference_steps
        req.strength = strength
        req.prompt = req.prompt
        req.image = upscaled_image

        pipe = self.resources["img2img"]

        if SD_USE_HYPERTILE:
            upscaled_image = hypertile(
                pipe, aspect_ratio=width / height, **req.__dict__
            ).images[0]
        else:
            upscaled_image = pipe(**req.__dict__).images[0]

        return upscaled_image


async def _handle_request(
    req: Txt2ImgRequest,
):
    plugin = None
    try:
        plugin: StableDiffusionPlugin = await use_plugin(StableDiffusionPlugin)
        image = get_image_from_request(req.image) if req.image else None
        mode = "img2img" if image else "txt2img"
        # TODO: inpaint if mask provided
        response = await plugin.generate(mode, req)
        return StableDiffusionPlugin.format_response(req, response)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin is not None:
            release_plugin(StableDiffusionPlugin)


@PluginBase.router.post("/txt2img", tags=["Image Generation (text-to-image)"])
async def txt2img(req: Txt2ImgRequest):
    return await _handle_request(req)


@PluginBase.router.get("/txt2img", tags=["Image Generation (text-to-image)"])
async def txt2img_get(
    req: Txt2ImgRequest = Depends(),
):
    return await _handle_request(req)


@PluginBase.router.post("/img2img", tags=["Image Generation (image-to-image)"])
async def img2img(req: Txt2ImgRequest):
    return await _handle_request(req)


@PluginBase.router.get("/img2img", tags=["Image Generation (image-to-image)"])
async def img2img_from_url(
    req: Txt2ImgRequest = Depends(),
):
    return await _handle_request(req)


@PluginBase.router.post("/inpaint", tags=["Image Generation (text-to-image)"])
async def inpaint(
    req: Txt2ImgRequest,
):
    plugin = None
    try:
        plugin: StableDiffusionPlugin = await use_plugin(StableDiffusionPlugin)
        input_image = get_image_from_request(req.image)
        response = await plugin.generate("inpaint", req, image=input_image)
        return StableDiffusionPlugin.format_response(req, response)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin is not None:
            release_plugin(StableDiffusionPlugin)


@PluginBase.router.get("/inpaint", tags=["Image Generation (text-to-image)"])
async def inpaint_from_url(
    req: Txt2ImgRequest = Depends(),
):
    plugin = None
    try:
        plugin: StableDiffusionPlugin = await use_plugin(StableDiffusionPlugin)
        input_image = get_image_from_request(req.image)
        response = await plugin.generate("inpaint", req, image=input_image)
        return StableDiffusionPlugin.format_response(req, response)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if plugin is not None:
            release_plugin(StableDiffusionPlugin)
