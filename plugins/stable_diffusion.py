import logging
import os
from typing import Literal, Optional

import diffusers
import torch
import tqdm.rich
from accelerate import cpu_offload
from diffusers import DiffusionPipeline
from fastapi import Depends
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import hf_hub_download
from PIL import Image

from classes.requests import ModelInfoRequest, Txt2ImgRequest
from modules.filter import filter_request
from modules.plugins import PluginBase, check_low_vram, release_plugin, use_plugin
from settings import (
    SD_COMPILE_UNET,
    SD_COMPILE_VAE,
    SD_DEFAULT_MODEL_INDEX,
    SD_HALF_VAE,
    SD_MIN_IMG2IMG_STEPS,
    SD_MODELS,
    SD_USE_DEEPCACHE,
    SD_USE_HYPERTILE,
    SD_USE_LIGHTNING_WEIGHTS,
    SD_USE_TOKEN_MERGING,
    SDXL_REFINER_MODEL,
    USE_XFORMERS,
)
from submodules.HiDiffusion.hidiffusion import apply_hidiffusion, remove_hidiffusion
from utils.console_logging import log_generate, log_loading, log_recycle
from utils.gpu_utils import autodetect_dtype, clear_gpu_cache, set_seed
from utils.hypertile_utils import hypertile
from utils.image_utils import (
    crop_and_resize,
    get_image_from_request,
    image_to_base64_no_header,
    image_to_bytes,
)
from utils.stable_diffusion_utils import (
    disable_freeu,
    enable_freeu,
    get_model,
    load_lora_settings,
    load_prompt_lora,
    manual_offload,
    postprocess,
)

YOLOS_MODEL = "hustvl/yolos-tiny"

IP_ADAPTERS = {
    "canny": "TencentARC/t2iadapter_canny_sd15v2",
    "cannyxl": "TencentARC/t2i-adapter-canny-sdxl-1.0",
    "qr": "monster-labs/control_v1p_sd15_qrcode_monster",
    "qrxl": "monster-labs/control_v1p_sdxl_qrcode_monster",
    "depth": "TencentARC/t2iadapter_depth_sd15v2",
    "depthxl": "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
}

if os.path.exists("models-sd.txt"):
    with open("models-sd.txt", "r") as f:
        models = f.read().splitlines()
        if models:
            SD_MODELS.clear()
            for model in models:
                if model not in SD_MODELS:
                    SD_MODELS.append(model)

else:
    with open("models-sd.txt", "w") as f:
        f.write("\n".join(SD_MODELS))


class StableDiffusionPlugin(PluginBase):
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )
    from diffusers.pipelines import (
        StableDiffusion3Img2ImgPipeline,
        StableDiffusion3Pipeline,
    )
    from nudenet import NudeDetector

    name = "Stable Diffusion"
    description = "Base plugin for txt2img, img2img, inpaint, etc."
    instance = None
    last_loras = []
    current_scheduler = None
    current_scheduler_name = None
    plugins = ["DetectYOLOSPlugin", "Txt2ImgFluxPlugin"]

    def __init__(
        self,
        pipeline_type=None,
        **model_kwargs,
    ):
        # diffusers.utils.logging.tqdm = tqdm.rich.tqdm

        # transformers.logging.set_verbosity_error()

        super().__init__()

        self.pipeline_type = pipeline_type
        self.dtype = autodetect_dtype(False)
        self.model_index = None
        self.model_kwargs = model_kwargs
        self.model_default_steps = 14
        self.tiling = False
        self.scheduler_config = None
        self.current_ip_adapter = None

    def offload(self):
        manual_offload(self.resources.get("pipeline"))

    def unload(self):
        # HACK: pipeline won't delete from VRAM unless this is enabled
        if self.resources.get("pipeline"):
            self.resources["pipeline"].enable_model_cpu_offload()

        if self.resources.get("txt2img"):
            del self.resources["txt2img"]
        if self.resources.get("img2img"):
            del self.resources["img2img"]
        if self.resources.get("inpaint"):
            del self.resources["inpaint"]

        if self.resources.get("pipeline"):
            self.resources["pipeline"].unload_lora_weights()
            self.resources["pipeline"].maybe_free_model_hooks()
            del self.resources["pipeline"]

        clear_gpu_cache()

    def load_ip_adapter(self, ip_adapter_type: str):
        ip_adapter = IP_ADAPTERS.get(ip_adapter_type)
        if ip_adapter and ip_adapter_type != self.current_ip_adapter:
            if self.resources.get("ip_adapter"):
                del self.resources["ip_adapter"]
                self.resources["ip_adapter"] = None
            log_loading("IP adapter", ip_adapter)
            from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

            self.resources["ip_adapter"] = StableDiffusionAdapterPipeline.from_pipe(
                self.resources.get("pipeline"),
                T2IAdapter.from_pretrained(
                    ip_adapter, torch_dtype=self.dtype, device=self.device
                ),
                requires_safety_checker=False,
            )

    def create_additional_pipelines(self, image_pipeline):
        if self.__class__ == StableDiffusionPlugin:
            from diffusers import (
                AutoPipelineForImage2Image,
                AutoPipelineForInpainting,
                AutoPipelineForText2Image,
                StableDiffusionAdapterPipeline,
                StableDiffusionXLAdapterPipeline,
            )

            is_sdxl = "XL" in image_pipeline.__class__.__name__
            is_sd3 = "StableDiffusion3" in image_pipeline.__class__.__name__

            adapter_type = (
                StableDiffusionXLAdapterPipeline
                if is_sdxl
                else StableDiffusionAdapterPipeline
            )

            pipe_kwargs = dict(
                pipeline=image_pipeline, device=self.device, dtype=self.dtype
            )

            if not is_sdxl and not is_sd3:
                pipe_kwargs["requires_safety_checker"] = False

            self.resources["txt2img"] = self.resources.get(
                "txt2img", AutoPipelineForText2Image.from_pipe(**pipe_kwargs)
            )
            self.resources["img2img"] = self.resources.get(
                "img2img", AutoPipelineForImage2Image.from_pipe(**pipe_kwargs)
            )
            self.resources["inpaint"] = self.resources.get(
                "inpaint", AutoPipelineForInpainting.from_pipe(**pipe_kwargs)
            )

    def load_model(self, model_index: int = SD_DEFAULT_MODEL_INDEX, **pipeline_kwargs):
        if model_index != self.model_index:
            from diffusers import (
                AutoencoderKL,
                StableDiffusion3Pipeline,
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
            )

            lname = SD_MODELS[model_index].lower()

            pipeline_type = self.pipeline_type or (
                StableDiffusionXLPipeline
                if "xl" in lname
                else (
                    StableDiffusion3Pipeline
                    if "stable-diffusion-3" in lname
                    else StableDiffusionPipeline
                )
            )

            logging.info(f"Using pipeline: {pipeline_type.__name__}")

            is_sd3 = "sd3" in lname or "stable-diffusion-3" in lname
            is_lightning = not is_sd3 and "lightning" in lname
            is_turbo = not is_sd3 and "turbo" in lname
            is_sdxl = not is_sd3 and "xl" in lname

            self.model_default_steps = 10 if is_lightning else 14 if is_turbo else 25

            repo_or_path = SD_MODELS[model_index]

            if self.resources.get("pipeline") is not None:
                self.unload()

            model_path: str = get_model(repo_or_path)

            log_loading("image model", os.path.basename(model_path))

            single_file = repo_or_path.endswith(".safetensors")

            kwargs = {"torch_dtype": self.dtype}

            if not is_sdxl and not is_sd3:
                kwargs["requires_safety_checker"] = False

            if is_sdxl:
                if SD_HALF_VAE and self.dtype != torch.float32:
                    kwargs["vae"] = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix",
                        torch_dtype=self.dtype,
                        use_safetensors=True,
                        device=self.device,
                    )

            elif is_sd3:
                kwargs["text_encoder_3"] = None
                kwargs["torch_dtype"] = self.dtype

            from_model = (
                pipeline_type.from_single_file
                if single_file
                else pipeline_type.from_pretrained
            )

            image_pipeline: (
                StableDiffusionPipeline
                | StableDiffusionXLPipeline
                | StableDiffusion3Pipeline
            ) = from_model(
                model_path,
                **kwargs,
                **self.model_kwargs,
                **pipeline_kwargs,
            )

            self.create_additional_pipelines(image_pipeline)
            self.resources["pipeline"] = image_pipeline

            self.model_index = model_index
            self.last_loras = []

            if "lightning" in model_path.lower() and SD_USE_LIGHTNING_WEIGHTS:
                log_loading(
                    "LoRA",
                    "ByteDance/SDXL-Lightning/sdxl_lightning_8step_lora.safetensors",
                )
                image_pipeline.load_lora_weights(
                    hf_hub_download(
                        "ByteDance/SDXL-Lightning",
                        "sdxl_lightning_8step_lora.safetensors",
                    )
                )
                image_pipeline.fuse_lora()
                image_pipeline.unload_lora_weights()        
        
            if SD_COMPILE_UNET:
                image_pipeline.unet = torch.compile(
                    image_pipeline.unet, mode="reduce-overhead", fullgraph=True
                )
            if SD_COMPILE_VAE:
                image_pipeline.vae = torch.compile(
                    image_pipeline.vae, mode="reduce-overhead", fullgraph=True
                )

            if SD_USE_DEEPCACHE and not is_sd3 and not is_sdxl:
                from DeepCache import DeepCacheSDHelper

                helper = DeepCacheSDHelper(pipe=image_pipeline)
                helper.set_params(
                    cache_interval=3,
                    cache_branch_id=0,
                )
                helper.enable()
                self.resources["DeepCacheSDHelper"] = helper

            # This is now enabled by default in Pytorch 2.x
            # from diffusers.models.attention_processor import AttnProcessor2_0
            # image_pipeline.unet.set_attn_processor(AttnProcessor2_0())

            if SD_USE_TOKEN_MERGING:
                import tomesd

                tomesd.apply_patch(image_pipeline, ratio=0.5)

            if USE_XFORMERS and not is_sd3:
                image_pipeline.enable_xformers_memory_efficient_attention()
                image_pipeline.vae.enable_xformers_memory_efficient_attention()

            if not is_sd3 and not SD_USE_HYPERTILE:
                image_pipeline.enable_vae_slicing()
                image_pipeline.enable_vae_tiling()

            if not is_sd3:
                image_pipeline.unet.to(memory_format=torch.channels_last)

            if is_sd3:
                self.resources["txt2img"] = image_pipeline

            self.tiling = False
        else:
            log_recycle(f"Reusing image model: {SD_MODELS[model_index]}")
            image_pipeline = self.resources["pipeline"]

        self.resources["lora_settings"] = load_lora_settings(
            ""
            if image_pipeline.__class__.__name__ == "StableDiffusionXLPipeline"
            else "sd15"
        )
        image_pipeline.to(self.device, dtype=self.dtype)
        self.default_scheduler = image_pipeline.scheduler

        if not self.resources.get("AutoImageProcessor"):
            from transformers import AutoImageProcessor

            self.resources["AutoImageProcessor"] = AutoImageProcessor.from_pretrained(
                YOLOS_MODEL,
            )

        if not self.resources.get("AutoModelForObjectDetection"):
            from transformers import AutoModelForObjectDetection

            self.resources["AutoModelForObjectDetection"] = (
                AutoModelForObjectDetection.from_pretrained(
                    YOLOS_MODEL,
                )
            )

        if not self.resources.get("NudeDetector"):
            from nudenet import NudeDetector

            self.resources["NudeDetector"] = NudeDetector()

    def _get_refiner(self):
        if self.resources.get("refiner"):
            refiner = self.resources["refiner"]
            log_recycle(f"Reusing refiner: {SDXL_REFINER_MODEL}")
            return refiner
        else:
            image_pipeline = self.resources["pipeline"]

            if not image_pipeline:
                logging.warning("No pipeline found. Refiner not loaded.")
                return None

            log_loading("refiner", SDXL_REFINER_MODEL)

            refiner = DiffusionPipeline.from_pretrained(
                SDXL_REFINER_MODEL,
                # unet=custom_unet,
                text_encoder_2=image_pipeline.text_encoder_2,
                vae=image_pipeline.vae,
                torch_dtype=self.dtype,
                use_safetensors=True,
                variant="fp16",
            )
            refiner.progress_bar = tqdm.rich.tqdm
            refiner.scheduler = image_pipeline.scheduler

            refiner.enable_model_cpu_offload(None, self.device)

            self.resources["refiner"] = refiner

            return refiner

    def get_scheduler(self, name: str):
        from diffusers import (
            DDIMScheduler,
            DDPMScheduler,
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            HeunDiscreteScheduler,
            KDPM2AncestralDiscreteScheduler,
            KDPM2DiscreteScheduler,
            LMSDiscreteScheduler,
            TCDScheduler,
        )

        if name is None:
            logging.info(
                f"Using default scheduler: {self.default_scheduler.__class__.__name__}"
            )
            return self.default_scheduler

        if name == self.current_scheduler_name:
            return self.current_scheduler

        default_config = self.default_scheduler.config.copy()

        if name == "euler":
            scheduler = EulerDiscreteScheduler.from_config(default_config)
        elif name == "euler_a":
            scheduler = EulerAncestralDiscreteScheduler.from_config(default_config)
        elif name == "lms":
            scheduler = LMSDiscreteScheduler.from_config(default_config)
        elif name == "heun":
            scheduler = HeunDiscreteScheduler.from_config(default_config)
        elif name == "ddim":
            scheduler = DDIMScheduler.from_config(default_config)
        elif name == "ddpm":
            scheduler = DDPMScheduler.from_config(default_config)
        elif name == "kdpm":
            scheduler = KDPM2DiscreteScheduler.from_config(
                default_config, use_karras_sigmas=True
            )
        elif name == "kdpm_a":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(default_config)
        elif name == "tcd":
            scheduler = TCDScheduler.from_config(default_config)
        elif name == "tcd_test":
            scheduler = TCDScheduler()
        elif name == "dpm":
            scheduler = DPMSolverSinglestepScheduler.from_config(default_config)
        elif name == "dpm2m":
            scheduler = DPMSolverMultistepScheduler.from_config(default_config)
        elif name == "sde":
            scheduler = DPMSolverMultistepScheduler.from_config(
                default_config, algorithm_type="sde-dpmsolver++"
            )
        elif name == "sde_karras":
            scheduler = DPMSolverMultistepScheduler.from_config(
                default_config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
            )
        else:
            raise ValueError(f"Invalid scheduler name: {name}")

        self.current_scheduler_name = name
        self.current_scheduler = scheduler

        karras_str = (
            " (Karras)" if scheduler.config.get("use_karras_sigmas", False) else ""
        )
        logging.info(f"Using scheduler: {scheduler.__class__.__name__}{karras_str}")

        return scheduler

    async def generate(
        self,        
        req: Txt2ImgRequest,
        **external_kwargs,
    ):
        req = filter_request(req)
        self.load_model(req.model_index)
        image_pipeline = self.resources["pipeline"]

        if req.tiling != self.tiling:
            set_tiling(image_pipeline, req.tiling, req.tiling)
            self.tiling = req.tiling

        if req.freeu:
            enable_freeu(image_pipeline)
        else:
            disable_freeu(image_pipeline)

        lname = SD_MODELS[req.model_index].lower()
        is_sd3 = "sd3" in lname or "stable-diffusion-3" in lname
        is_xl = "xl" in lname

        if not is_sd3:
            if req.hi:
                # HiDiffusion expects this property to be set
                # https://github.com/megvii-research/HiDiffusion/issues/26#issuecomment-2365181245
                setattr(image_pipeline, "_num_timesteps", req.num_inference_steps)
                apply_hidiffusion(image_pipeline)
            else:
                remove_hidiffusion(image_pipeline)

        if not is_sd3:
            scheduler = self.get_scheduler(req.scheduler)

            if not scheduler:
                scheduler = self.default_scheduler

            scheduler.config["lower_order_final"] = not is_xl

            image_pipeline.scheduler = scheduler

        if self.resources.get("inpaint"):
            self.resources["inpaint"].scheduler = image_pipeline.scheduler

        if req.auto_lora:
            lora_settings = self.resources["lora_settings"]

            self.last_loras = load_prompt_lora(
                image_pipeline, req, lora_settings, self.last_loras
            )

        if req.image is not None and req.num_inference_steps == 0:
            image = get_image_from_request(req.image, (req.width, req.height))
            pass

        else:
            mode = "ip_adapter" if req.adapter else "inpaint" if req.mask_image else "img2img" if req.image else "txt2img"
            pipe: DiffusionPipeline = self.resources.get(mode, image_pipeline)

            pipe.progress_bar = tqdm.rich.tqdm

            pipe.scheduler = image_pipeline.scheduler

            req.seed, generator = set_seed(req.seed, True)

            args = dict(
                prompt=req.prompt,
                width=req.width,
                height=req.height,
                num_images_per_prompt=req.num_images_per_prompt,
                guidance_scale=req.guidance_scale,
                num_inference_steps=req.num_inference_steps,
                generator=generator,
            )

            result = None

            if (
                req.image is not None
                and pipe.__class__.__name__ != "StableDiffusionXLAdapterPipeline"
            ):
                args["strength"] = req.strength

            if not is_xl and not is_sd3:
                args["requires_safety_checker"] = False

            if req.adapter:
                if is_xl:
                    req.adapter += "xl"
                self.load_ip_adapter(req.adapter)

            # if req.upscale >= 1:
            #     args["output_type"] = "latent"

            if req.use_refiner and is_xl:
                args["output_type"] = "latent"
                args["denoising_end"] = 0.8

            if req.negative_prompt:
                args["negative_prompt"] = req.negative_prompt

            if req.image is not None:
                image = get_image_from_request(req.image, (req.width, req.height))
                args["image"] = [image]
                logging.info(
                    f"Using provided image as input for {req.adapter or mode}."
                )

            log_generate(f"Generating image ({req.width}x{req.height})...")
            if SD_USE_HYPERTILE:
                result = hypertile(pipe, **args, **external_kwargs)
            else:
                result = pipe(**args, **external_kwargs)

            if req.hi:
                remove_hidiffusion(image_pipeline)           

            if req.use_refiner and is_xl:
                refiner = self._get_refiner()

                log_generate("Refining image(s)...")

                result.images = [
                    refiner(
                        prompt=args["prompt"],
                        num_inference_steps=args["num_inference_steps"],
                        denoising_start=0.8,
                        image=image,
                    ).images[0]
                    for image in result.images
                ]

            pipe.maybe_free_model_hooks()

        if self.__class__ == StableDiffusionPlugin:
            images, json_response = await postprocess(
                self, result.images if result else req.image, req, **external_kwargs
            )
            result.images = images
        else:
            json_response = { "seed": req.seed, "images": [image_to_base64_no_header(img) for img in result.images] }

        if req.return_json:
            return json_response

        check_low_vram()

        return result.images[0]

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

        if num_inference_steps * req.strength < SD_MIN_IMG2IMG_STEPS:
            logging.warn("Increasing steps to prevent artifacts")
            num_inference_steps = int(6 // req.strength)

        if req.strength > 1:
            logging.warn(f"Limiting strength to 2 from {req.strength}")
            strength = 1
        else:
            strength = req.strength

        req.width = req.width or image.width
        req.height = req.height or image.height

        width = int((req.width * req.upscale) // 128 * 128)
        height = int((req.height * req.upscale) // 128 * 128)

        upscaled_image = crop_and_resize(image, (width, height))

        _, generator = set_seed(req.seed, True)

        req.num_inference_steps = num_inference_steps
        req.strength = strength
        req.image = upscaled_image

        pipe = self.resources["img2img"]

        req = req.__dict__
        req["generator"] = generator

        if SD_USE_HYPERTILE:
            req["aspect_ratio"] = width / height
            upscaled_image = hypertile(pipe, **req).images[0]
        else:
            upscaled_image = pipe(**req).images[0]

        return upscaled_image


async def _handle_request(
    req: Txt2ImgRequest,
):
    plugin: StableDiffusionPlugin = None
    try:
        plugin = await use_plugin(StableDiffusionPlugin)
        image = get_image_from_request(req.image) if req.image else None

        if image:
            if not req.width:
                req.width = image.size[0]
            if not req.height:
                req.height = image.size[1]

        mode = "inpaint" if req.mask_image else "ip_adapter" if req.adapter else "img2img" if req.image else "txt2img"
        
        response = await plugin.generate(req)

        return format_response(response)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin is not None:
            release_plugin(StableDiffusionPlugin)


@PluginBase.router.post("/txt2img", tags=["Image Generation"])
async def txt2img(req: Txt2ImgRequest):
    return await _handle_request(req)


@PluginBase.router.get("/txt2img", tags=["Image Generation"])
async def txt2img_get(
    req: Txt2ImgRequest = Depends(),
):
    return await _handle_request(req)


@PluginBase.router.post("/img2img", tags=["Image Generation"])
async def img2img(req: Txt2ImgRequest):
    return await _handle_request(req)


@PluginBase.router.get("/img2img", tags=["Image Generation"])
async def img2img_from_url(
    req: Txt2ImgRequest = Depends(),
):
    return await _handle_request(req)


@PluginBase.router.post("/inpaint", tags=["Image Generation"])
async def inpaint(
    req: Txt2ImgRequest,
):
    plugin = None
    try:
        plugin: StableDiffusionPlugin = await use_plugin(StableDiffusionPlugin)
        input_image = get_image_from_request(req.image)
        response = await plugin.generate(req, image=input_image)
        return format_response(response)
    except Exception as e:
        logging.error(e, exc_info=True)
        raise e
    finally:
        if plugin is not None:
            release_plugin(StableDiffusionPlugin)


@PluginBase.router.get("/inpaint", tags=["Image Generation"])
async def inpaint_from_url(
    req: Txt2ImgRequest = Depends(),
):
    return await inpaint(req)


def set_tiling(pipeline, x_axis, y_axis):
    """Utility function used to configure the pipeline to generate seamless images.
    Thanks to alexisrolland, https://github.com/huggingface/diffusers/issues/556#issuecomment-1968455622
    """

    from diffusers.models.lora import LoRACompatibleConv

    def asymmetric_conv2d_convforward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        self.paddingX = (
            self._reversed_padding_repeated_twice[0],
            self._reversed_padding_repeated_twice[1],
            0,
            0,
        )
        self.paddingY = (
            0,
            0,
            self._reversed_padding_repeated_twice[2],
            self._reversed_padding_repeated_twice[3],
        )
        working = torch.nn.functional.pad(input, self.paddingX, mode=x_mode)
        working = torch.nn.functional.pad(working, self.paddingY, mode=y_mode)
        return torch.nn.functional.conv2d(
            working,
            weight,
            bias,
            self.stride,
            torch.nn.modules.utils._pair(0),
            self.dilation,
            self.groups,
        )

    # Set padding mode
    x_mode = "circular" if x_axis else "constant"
    y_mode = "circular" if y_axis else "constant"

    # For SDXL models
    is_xl = "XL" in pipeline.__class__.__name__
    if is_xl:
        targets = [
            pipeline.vae,
            pipeline.text_encoder,
            pipeline.text_encoder_2,
            pipeline.unet,
        ]

    # For SD1.5 and SD2.1 models
    else:
        targets = [pipeline.vae, pipeline.text_encoder, pipeline.unet]

    convolution_layers = []
    for target in targets:
        for module in target.modules():
            if isinstance(module, torch.nn.Conv2d):
                convolution_layers.append(module)

    for layer in convolution_layers:
        if isinstance(layer, LoRACompatibleConv) and layer.lora_layer is None:
            layer.lora_layer = lambda *x: 0

        layer._conv_forward = asymmetric_conv2d_convforward.__get__(
            layer, torch.nn.Conv2d
        )

    return pipeline


def format_response(response):
    if isinstance(response, dict):        
        return JSONResponse(response)

    return StreamingResponse(
        image_to_bytes(response),
        media_type="image/png",
    )


@PluginBase.router.post("/txt2img/model_info", tags=["System Info"])
async def model_info(req: ModelInfoRequest):
    if req.model_index:
        model_path = SD_MODELS[req.model_index]
        return JSONResponse(
            {"model_name": os.path.basename(model_path).rstrip(".safetensors")}
        )
    else:
        return JSONResponse(
            {"models": [os.path.basename(f).rstrip(".safetensors") for f in SD_MODELS]}
        )


@PluginBase.router.get("/txt2img/model_info", tags=["System Info"])
async def model_info_get(req: ModelInfoRequest = Depends()):
    return await model_info(req)
