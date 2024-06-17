import logging
import math
from fastapi import Depends
import numpy as np
import torch
from classes.requests import Txt2ImgRequest
from modules.filter import filter_request
from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.stable_diffusion import format_response
from utils.image_utils import get_image_from_request
from utils.stable_diffusion_utils import get_model, postprocess
from PIL import Image
from enum import Enum
from nudenet import NudeDetector
import safetensors.torch as sf
from huggingface_hub import hf_hub_download


class BGSource(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"


class Txt2ImgRelightPlugin(PluginBase):

    name: str = "txt2img_relight"
    description: str = "Text-to-image relighting plugin"
    instance: None

    def __init__(self):
        super().__init__()
        self.current_model_name = None

    def load_model(self, model_name="SG161222/Realistic_Vision_V5.1_noVAE"):

        if model_name == self.current_model_name:
            return

        self.current_model_name = model_name

        from transformers import CLIPTokenizer, CLIPTextModel
        from diffusers import (
            AutoencoderKL,
            UNet2DConditionModel,
            StableDiffusionPipeline,
            StableDiffusionImg2ImgPipeline,
            DPMSolverMultistepScheduler,
        )

        model_path = get_model(model_name)

        tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )
        self.resources["tokenizer"] = tokenizer

        text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )
        self.resources["text_encoder"] = text_encoder

        vae: AutoencoderKL = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
        self.resources["vae"] = vae

        unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet"
        )

        with torch.no_grad():
            new_conv_in = torch.nn.Conv2d(
                8,
                unet.conv_in.out_channels,
                unet.conv_in.kernel_size,
                unet.conv_in.stride,
                unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            new_conv_in.bias = unet.conv_in.bias
            unet.conv_in = new_conv_in

        unet_original_forward = unet.forward

        def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
            c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
            c_concat = torch.cat(
                [c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0
            )
            new_sample = torch.cat([sample, c_concat], dim=1)
            kwargs["cross_attention_kwargs"] = {}
            return unet_original_forward(
                new_sample, timestep, encoder_hidden_states, **kwargs
            )

        unet.forward = hooked_unet_forward

        try:
            from submodules.IC_Light.briarmbg import BriaRMBG

            ic_light_path = hf_hub_download(
                "lllyasviel/ic-light", "iclight_sd15_fc.safetensors"
            )
        except Exception as e:
            logging.error(f"Error downloading ic-light: {str(e)}")
            raise e

        sd_offset = sf.load_file(ic_light_path)
        sd_origin = unet.state_dict()
        keys = sd_origin.keys()
        sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
        unet.load_state_dict(sd_merged, strict=True)
        del sd_offset, sd_origin, sd_merged, keys

        unet = unet.to(self.device, dtype=self.dtype)

        self.resources["unet"] = unet

        self.resources["scheduler"] = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1,
        )

        self.resources["txt2img"] = StableDiffusionPipeline(
            vae=self.resources["vae"],
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=self.resources["unet"],
            scheduler=self.resources["scheduler"],
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        ).to(self.device, dtype=self.dtype)

        self.resources["img2img"] = StableDiffusionImg2ImgPipeline(
            vae=self.resources["vae"],
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=self.resources["unet"],
            scheduler=self.resources["scheduler"],
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None,
            image_encoder=None,
        ).to(self.device, dtype=self.dtype)

        self.resources["rmbg"] = BriaRMBG.from_pretrained("briaai/RMBG-1.4").to(
            self.device
        )

        self.resources["NudeDetector"] = NudeDetector()

    @torch.inference_mode()
    def encode_prompt_inner(self, txt: str):

        tokenizer = self.resources["tokenizer"]
        text_encoder = self.resources["text_encoder"]

        max_length = tokenizer.model_max_length
        chunk_length = tokenizer.model_max_length - 2
        id_start = tokenizer.bos_token_id
        id_end = tokenizer.eos_token_id
        id_pad = id_end

        def pad(x, p, i):
            return x[:i] if len(x) >= i else x + [p] * (i - len(x))

        tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [
            [id_start] + tokens[i : i + chunk_length] + [id_end]
            for i in range(0, len(tokens), chunk_length)
        ]
        chunks = [pad(ck, id_pad, max_length) for ck in chunks]

        token_ids = torch.tensor(chunks).to(device=self.device, dtype=torch.int64)
        conds = text_encoder(token_ids).last_hidden_state

        return conds

    @torch.inference_mode()
    def encode_prompt_pair(self, positive_prompt, negative_prompt):
        c = self.encode_prompt_inner(positive_prompt)
        uc = self.encode_prompt_inner(negative_prompt)

        c_len = float(len(c))
        uc_len = float(len(uc))
        max_count = max(c_len, uc_len)
        c_repeat = int(math.ceil(max_count / c_len))
        uc_repeat = int(math.ceil(max_count / uc_len))
        max_chunk = max(len(c), len(uc))

        c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
        uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

        c = torch.cat([p[None, ...] for p in c], dim=1)
        uc = torch.cat([p[None, ...] for p in uc], dim=1)

        return c, uc

    @torch.inference_mode()
    def pytorch2numpy(self, imgs, quant=True):
        results = []
        for x in imgs:
            y = x.movedim(0, -1)

            if quant:
                y = y * 127.5 + 127.5
                y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
            else:
                y = y * 0.5 + 0.5
                y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

            results.append(y)
        return results

    @torch.inference_mode()
    def numpy2pytorch(self, imgs):
        h = (
            torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
        )  # so that 127 must be strictly 0.0
        h = h.movedim(-1, 1)
        return h

    def resize_and_center_crop(self, image, target_width, target_height):
        pil_image = Image.fromarray(image)
        original_width, original_height = pil_image.size
        scale_factor = max(
            target_width / original_width, target_height / original_height
        )
        resized_width = int(round(original_width * scale_factor))
        resized_height = int(round(original_height * scale_factor))
        resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
        left = (resized_width - target_width) / 2
        top = (resized_height - target_height) / 2
        right = (resized_width + target_width) / 2
        bottom = (resized_height + target_height) / 2
        cropped_image = resized_image.crop((left, top, right, bottom))
        return np.array(cropped_image)

    def resize_without_crop(self, image, target_width, target_height):
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
        return np.array(resized_image)

    @torch.inference_mode()
    def run_rmbg(self, img, sigma=0.0):

        from submodules.IC_Light.briarmbg import BriaRMBG

        rmbg: BriaRMBG = self.resources["rmbg"]

        H, W, C = img.shape
        assert C == 3
        k = (256.0 / float(H * W)) ** 0.5
        feed = self.resize_without_crop(
            img, int(64 * round(W * k)), int(64 * round(H * k))
        )
        feed = self.numpy2pytorch([feed]).to(device=self.device, dtype=torch.float32)
        alpha = rmbg(feed)[0][0]
        alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
        alpha = alpha.movedim(1, -1)[0]
        alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
        result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
        return result.clip(0, 255).astype(np.uint8), alpha

    @torch.inference_mode()
    def process(
        self,
        input_fg,
        prompt,
        image_width,
        image_height,
        num_samples,
        seed,
        steps,
        a_prompt,
        n_prompt,
        cfg,
        highres_scale,
        highres_denoise,
        lowres_denoise,
        bg_source,
    ):

        from diffusers import AutoencoderKL, UNet2DConditionModel

        unet: UNet2DConditionModel = self.resources["unet"]
        vae: AutoencoderKL = self.resources["vae"]
        t2i_pipe = self.resources["txt2img"]
        i2i_pipe = self.resources["img2img"]

        bg_source = BGSource(bg_source)
        input_bg = None

        if bg_source == BGSource.NONE:
            pass
        elif bg_source == BGSource.LEFT:
            gradient = np.linspace(255, 0, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.RIGHT:
            gradient = np.linspace(0, 255, image_width)
            image = np.tile(gradient, (image_height, 1))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.TOP:
            gradient = np.linspace(255, 0, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        elif bg_source == BGSource.BOTTOM:
            gradient = np.linspace(0, 255, image_height)[:, None]
            image = np.tile(gradient, (1, image_width))
            input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
        else:
            raise "Wrong initial latent!"

        rng = torch.Generator(device=self.device).manual_seed(int(seed))

        fg = self.resize_and_center_crop(input_fg, image_width, image_height)

        concat_conds = self.numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = (
            vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
        )

        conds, unconds = self.encode_prompt_pair(
            positive_prompt=prompt + ", " + a_prompt, negative_prompt=n_prompt
        )

        if input_bg is None:
            latents = (
                t2i_pipe(
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    width=image_width,
                    height=image_height,
                    num_inference_steps=steps,
                    num_images_per_prompt=num_samples,
                    generator=rng,
                    output_type="latent",
                    guidance_scale=cfg,
                    cross_attention_kwargs={"concat_conds": concat_conds},
                ).images.to(vae.dtype)
                / vae.config.scaling_factor
            )
        else:
            bg = self.resize_and_center_crop(input_bg, image_width, image_height)
            bg_latent = self.numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
            bg_latent = (
                vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
            )
            latents = (
                i2i_pipe(
                    image=bg_latent,
                    strength=lowres_denoise,
                    prompt_embeds=conds,
                    negative_prompt_embeds=unconds,
                    width=image_width,
                    height=image_height,
                    num_inference_steps=int(round(steps / lowres_denoise)),
                    num_images_per_prompt=num_samples,
                    generator=rng,
                    output_type="latent",
                    guidance_scale=cfg,
                    cross_attention_kwargs={"concat_conds": concat_conds},
                ).images.to(vae.dtype)
                / vae.config.scaling_factor
            )

        pixels = vae.decode(latents).sample
        pixels = self.pytorch2numpy(pixels)
        pixels = [
            self.resize_without_crop(
                image=p,
                target_width=int(round(image_width * highres_scale / 64.0) * 64),
                target_height=int(round(image_height * highres_scale / 64.0) * 64),
            )
            for p in pixels
        ]

        pixels = self.numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
        latents = latents.to(device=unet.device, dtype=unet.dtype)

        image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8

        fg = self.resize_and_center_crop(input_fg, image_width, image_height)
        concat_conds = self.numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
        concat_conds = (
            vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
        )

        latents = (
            i2i_pipe(
                image=latents,
                strength=highres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=int(round(steps / highres_denoise)),
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )

        pixels = vae.decode(latents).sample

        return self.pytorch2numpy(pixels)

    @torch.inference_mode()
    def process_relight(
        self,
        input_fg,
        prompt,
        image_width,
        image_height,
        num_samples,
        seed,
        steps,
        a_prompt,
        n_prompt,
        cfg,
        highres_scale,
        highres_denoise,
        lowres_denoise,
        bg_source,
    ):
        input_fg, matting = self.run_rmbg(input_fg)
        results = self.process(
            input_fg,
            prompt,
            image_width,
            image_height,
            num_samples,
            seed,
            steps,
            a_prompt,
            n_prompt,
            cfg,
            highres_scale,
            highres_denoise,
            lowres_denoise,
            bg_source,
        )
        return results


@PluginBase.router.post("/txt2img/relight")
async def txt2img_relight(req: Txt2ImgRequest):
    plugin: Txt2ImgRelightPlugin = None
    try:
        fg_image = get_image_from_request(req.image)
        req.width = fg_image.width
        req.height = fg_image.height
        req = filter_request(req)

        plugin = await use_plugin(Txt2ImgRelightPlugin)
        plugin.load_model()
        result = plugin.process_relight(
            np.array(fg_image),
            req.prompt,
            req.width,
            req.height,
            1,
            req.seed,
            req.num_inference_steps,
            req.prompt,
            ",".join([req.negative_prompt, "dark"]),
            req.guidance_scale,
            req.upscale if req.upscale else 1.0,
            0.5,
            0.9,
            BGSource.NONE,
        )[0]

        req.upscale = 0
        image, json_response = await postprocess(plugin, Image.fromarray(result), req)
        return format_response(req, json_response)

    except Exception as e:
        logging.error(f"Error in txt2img_relight: {str(e)}", exc_info=True)
        return {"error": str(e)}

    finally:
        if plugin:
            release_plugin(Txt2ImgRelightPlugin)


@PluginBase.router.get("/txt2img/relight")
async def txt2img_relight_from_url(req: Txt2ImgRequest = Depends()):
    return await txt2img_relight(req)
