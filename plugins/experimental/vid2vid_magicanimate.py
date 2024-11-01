from collections import OrderedDict
import datetime
import inspect
import os
import safetensors.torch
from PIL import Image
import cv2
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from einops import rearrange
import safetensors
import torch
from modules.plugins import PluginBase, use_plugin_unsafe
from plugins.txt2vid_animate import CLIP_MODELS
from settings import CACHE_PATH, SD_MODELS, USE_XFORMERS
from utils.console_logging import log_loading
from utils.file_utils import cached_snapshot
from utils.gpu_utils import set_seed
from utils.image_utils import crop_and_resize, get_image_from_request
from utils.stable_diffusion_utils import get_model
from submodules.MagicAnimate.magicanimate.utils.videoreader import VideoReader


IMAGE_ORDERING = "t h w c -> 1 c t h w"


class Vid2VidMagicAnimatePlugin(PluginBase):
    name = "MagicAnimate"
    description = "Motion transfer with magic animate"
    instance = None
    plugins = ["StableDiffusionPlugin"]

    def __init__(self):
        super().__init__()
        from omegaconf import OmegaConf
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        from submodules.MagicAnimate.magicanimate.pipelines.pipeline_animation import (
            AnimationPipeline,
        )
        from submodules.MagicAnimate.magicanimate.models.controlnet import (
            ControlNetModel,
        )
        from submodules.MagicAnimate.magicanimate.models.appearance_encoder import (
            AppearanceEncoderModel,
        )
        from submodules.MagicAnimate.magicanimate.models.mutual_self_attention import (
            ReferenceAttentionControl,
        )
        from submodules.MagicAnimate.magicanimate.models.unet_controlnet import (
            UNet3DConditionModel,
        )
        # from diffusers import StableDiffusionPipeline

        config = OmegaConf.load(
            "submodules/MagicAnimate/configs/prompts/animation.yaml"
        )

        inference_config = OmegaConf.load(config.inference_config)

        unet_additional_kwargs = OmegaConf.to_container(
            inference_config.unet_additional_kwargs
        )

        # model = cached_snapshot("SG161222/Realistic_Vision_V5.1_noVAE", ["R*.safetensors", "*.ckpt"])

        model = SD_MODELS[4]
        log_loading("model", model)
        state_dict = safetensors.torch.load_file(model, device="cpu")
        unet = UNet3DConditionModel.from_state_dict(
            state_dict, unet_additional_kwargs
        ).to(self.dtype)
        unet.config.sample_size = 64

        sd_pipeline: StableDiffusionPipeline = StableDiffusionPipeline.from_single_file(
            model, unet=unet, device=self.device, torch_dtype=self.dtype
        ).to(device=self.device, dtype=self.dtype)

        controlnet_model_path = cached_snapshot("zcxu-eric/MagicAnimate")
        motion_module = os.path.join(
            controlnet_model_path, "temporal_attention", "temporal_attention.ckpt"
        )
        pretrained_controlnet_path = os.path.join(
            controlnet_model_path, "densepose_controlnet"
        )
        pretrained_appearance_encoder_path = os.path.join(
            controlnet_model_path, "appearance_encoder"
        )

        appearance_encoder: AppearanceEncoderModel = (
            AppearanceEncoderModel.from_pretrained(
                pretrained_appearance_encoder_path, subfolder="appearance_encoder"
            ).cuda()
        )
        appearance_encoder.to(self.dtype)
        self.resources["appearance_encoder"] = appearance_encoder

        controlnet: ControlNetModel = ControlNetModel.from_pretrained(
            pretrained_controlnet_path
        )
        controlnet.to(self.dtype)
        self.resources["controlnet"] = controlnet

        if USE_XFORMERS:
            self.appearance_encoder.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()

        pipeline: AnimationPipeline = AnimationPipeline(
            unet=unet,
            vae=sd_pipeline.vae,
            text_encoder=sd_pipeline.text_encoder,
            tokenizer=sd_pipeline.tokenizer,
            controlnet=controlnet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
            ),
            # NOTE: UniPCMultistepScheduler
        ).to(device=self.device, dtype=self.dtype)

        pipeline.enable_sequential_cpu_offload()

        self.resources["pipeline"] = pipeline

        self.resources["reference_control_writer"] = ReferenceAttentionControl(
            appearance_encoder,
            do_classifier_free_guidance=True,
            mode="write",
            fusion_blocks=config.fusion_blocks,
        )
        self.resources["reference_control_reader"] = ReferenceAttentionControl(
            pipeline.unet,
            do_classifier_free_guidance=True,
            mode="read",
            fusion_blocks=config.fusion_blocks,
        )

        *_, func_args = inspect.getargvalues(inspect.currentframe())

        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        if "global_step" in motion_module_state_dict:
            func_args.update({"global_step": motion_module_state_dict["global_step"]})
        motion_module_state_dict = (
            motion_module_state_dict["state_dict"]
            if "state_dict" in motion_module_state_dict
            else motion_module_state_dict
        )
        try:
            # extra steps for self-trained models
            state_dict = OrderedDict()
            for key in motion_module_state_dict.keys():
                if key.startswith("module."):
                    _key = key.split("module.")[-1]
                    state_dict[_key] = motion_module_state_dict[key]
                else:
                    state_dict[key] = motion_module_state_dict[key]
            motion_module_state_dict = state_dict
            del state_dict
            missing, unexpected = pipeline.unet.load_state_dict(
                motion_module_state_dict, strict=False
            )
            # assert len(unexpected) == 0
        except Exception:
            _tmp_ = OrderedDict()
            for key in motion_module_state_dict.keys():
                if "motion_modules" in key:
                    if key.startswith("unet."):
                        _key = key.split("unet.")[-1]
                        _tmp_[_key] = motion_module_state_dict[key]
                    else:
                        _tmp_[key] = motion_module_state_dict[key]
            missing, unexpected = pipeline.unet.load_state_dict(_tmp_, strict=False)
            # assert len(unexpected) == 0
            del _tmp_
        del motion_module_state_dict

        motion_module_state_dict = torch.load(motion_module, map_location="cpu")
        if "global_step" in motion_module_state_dict:
            func_args.update({"global_step": motion_module_state_dict["global_step"]})
        motion_module_state_dict = (
            motion_module_state_dict["state_dict"]
            if "state_dict" in motion_module_state_dict
            else motion_module_state_dict
        )
        try:
            # extra steps for self-trained models
            state_dict = OrderedDict()
            for key in motion_module_state_dict.keys():
                if key.startswith("module."):
                    _key = key.split("module.")[-1]
                    state_dict[_key] = motion_module_state_dict[key]
                else:
                    state_dict[key] = motion_module_state_dict[key]
            motion_module_state_dict = state_dict
            del state_dict
            missing, unexpected = pipeline.unet.load_state_dict(
                motion_module_state_dict, strict=False
            )
            assert len(unexpected) == 0
        except Exception:
            _tmp_ = OrderedDict()
            for key in motion_module_state_dict.keys():
                if "motion_modules" in key:
                    if key.startswith("unet."):
                        _key = key.split("unet.")[-1]
                        _tmp_[_key] = motion_module_state_dict[key]
                    else:
                        _tmp_[key] = motion_module_state_dict[key]
            missing, unexpected = pipeline.unet.load_state_dict(_tmp_, strict=False)
            # assert len(unexpected) == 0
            del _tmp_
        del motion_module_state_dict

        self.L = config.L

    def animate(
        self,
        image: Image.Image,
        motion_sequence,
        random_seed,
        step,
        guidance_scale,
        size=512,
    ):
        pipeline = self.resources["pipeline"]
        appearance_encoder = self.resources["appearance_encoder"]
        reference_control_writer = self.resources["reference_control_writer"]
        reference_control_reader = self.resources["reference_control_reader"]
        source_image = np.array(image)

        prompt = n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        samples_per_video = []
        # manually set random seed for reproduction
        if random_seed != -1:
            torch.manual_seed(random_seed)
            set_seed(random_seed)
        else:
            torch.seed()

        if motion_sequence.endswith(".mp4"):
            control = VideoReader(motion_sequence).read()
            if control[0].shape[0] != size:
                control = [
                    np.array(Image.fromarray(c).resize((size, size))) for c in control
                ]
            control = np.array(control)

        if source_image.shape[0] != size:
            source_image = np.array(Image.fromarray(source_image).resize((size, size)))
        H, W, C = source_image.shape

        init_latents = None
        original_length = control.shape[0]
        if control.shape[0] % self.L > 0:
            control = np.pad(
                control,
                ((0, self.L - control.shape[0] % self.L), (0, 0), (0, 0), (0, 0)),
                mode="edge",
            )
        generator = torch.Generator(device=torch.device("cuda:0"))
        generator.manual_seed(torch.initial_seed())
        sample = pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            video_length=len(control),
            controlnet_condition=control,
            init_latents=init_latents,
            generator=generator,
            appearance_encoder=appearance_encoder,
            reference_control_writer=reference_control_writer,
            reference_control_reader=reference_control_reader,
            source_image=source_image,
        ).videos

        source_images = np.array([source_image] * original_length)
        source_images = (
            rearrange(torch.from_numpy(source_images), IMAGE_ORDERING) / 255.0
        )
        samples_per_video.append(source_images)

        control = control / 255.0
        control = rearrange(control, IMAGE_ORDERING)
        control = torch.from_numpy(control)
        samples_per_video.append(control[:, :, :original_length])

        samples_per_video.append(sample[:, :, :original_length])

        samples_per_video = torch.cat(samples_per_video)

        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        savedir = CACHE_PATH
        animation_path = f"{savedir}/{time_str}.mp4"
        # TODO

        # os.makedirs(savedir, exist_ok=True)
        # save_videos_grid(samples_per_video, animation_path)

        return animation_path

    def generate(
        self,
        reference_image,
        motion_sequence_state,
        seed=1,
        steps=25,
        guidance_scale=7.5,
    ):
        from submodules.MagicAnimate.magicanimate.pipelines.pipeline_animation import (
            AnimationPipeline,
        )

        pipeline: AnimationPipeline = self.resources["pipeline"]
        if not pipeline:
            raise HTTPException(
                status_code=500,
                detail="Pipeline not loaded",
            )

        # Get image from request
        image: Image.Image = get_image_from_request(reference_image)

        size = min(image.width, image.height)
        image = crop_and_resize(image, (size, size))

        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        path = self.animate(
            image,
            motion_sequence_state,
            seed,
            steps,
            guidance_scale,
        )

        return StreamingResponse(
            path,
            media_type="video/mp4",
        )
