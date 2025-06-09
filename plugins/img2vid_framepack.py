import logging
from typing import Optional

import einops
import numpy as np
import torch
from diffusers import AutoencoderKLHunyuanVideo
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    LlamaModel,
    LlamaTokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from modules.plugins import PluginBase, use_plugin
from settings import CACHE_PATH
from submodules.FramePack.diffusers_helper.bucket_tools import find_nearest_bucket
from submodules.FramePack.diffusers_helper.clip_vision import hf_clip_vision_encode
from submodules.FramePack.diffusers_helper.gradio.progress_bar import (
    make_progress_bar_html,
)
from submodules.FramePack.diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_decode_fake,
    vae_encode,
)
from submodules.FramePack.diffusers_helper.memory import (
    DynamicSwapInstaller,
    cpu,
    fake_diffusers_current_device,
    get_cuda_free_memory_gb,
    gpu,
    load_model_as_complete,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    unload_complete_models,
)
from submodules.FramePack.diffusers_helper.models.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked,
)
from submodules.FramePack.diffusers_helper.pipelines.k_diffusion_hunyuan import (
    sample_hunyuan,
)
from submodules.FramePack.diffusers_helper.thread_utils import AsyncStream, async_run
from submodules.FramePack.diffusers_helper.utils import (
    crop_or_pad_yield_mask,
    generate_timestamp,
    resize_and_center_crop,
    save_bcthw_as_mp4,
    soft_append_bcthw,
    state_dict_offset_merge,
    state_dict_weighted_merge,
)
from utils.console_logging import log_generate, log_loading
from utils.file_utils import random_filename
from utils.gpu_utils import set_seed
from utils.image_utils import get_image_from_request


class Img2VidFramePackRequest(BaseModel):
    image: str
    prompt: str
    negative_prompt: Optional[str] = None
    length: Optional[float] = 5  # max 120    
    guidance_scale: Optional[float] = 32.0  # max 1
    num_inference_steps: Optional[int] = 35  # max 100
    seed: Optional[int] = -1
    latent_window_size : Optional[int] = 9  # max 33
    use_teacache: Optional[bool] = True # Faster, worse quality on hands etc


class Img2VidFramePackPlugin(PluginBase):
    name = "Image-to-Video (FramePack)"
    description = "Image-to-video using FramePack"
    instance = None

    def __init__(self):
        super().__init__()

    def load_model(self):
        if self.resources.get("transformer"):
            return

        log_loading("FramePack", "cache")

        free_mem_gb = get_cuda_free_memory_gb(gpu)
        high_vram = free_mem_gb > 60

        self.high_vram = high_vram

        print(f"Free VRAM {free_mem_gb} GB")
        print(f"High-VRAM Mode: {high_vram}")

        text_encoder = LlamaModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        ).cpu()
        text_encoder_2 = CLIPTextModel.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        ).cpu()
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2"
        )
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            subfolder="vae",
            torch_dtype=torch.float16,
        ).cpu()
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl",
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        ).cpu()
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
            "lllyasviel/FramePack_F1_I2V_HY_20250503", torch_dtype=torch.bfloat16
        ).cpu()

        self.resources["text_encoder"] = text_encoder
        self.resources["text_encoder_2"] = text_encoder_2
        self.resources["tokenizer"] = tokenizer
        self.resources["tokenizer_2"] = tokenizer_2
        self.resources["vae"] = vae
        self.resources["feature_extractor"] = feature_extractor
        self.resources["image_encoder"] = image_encoder
        self.resources["transformer"] = transformer

        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()

        if not high_vram:
            vae.enable_slicing()
            vae.enable_tiling()

        transformer.high_quality_fp32_output_for_inference = True
        print("transformer.high_quality_fp32_output_for_inference = True")

        transformer.to(dtype=torch.bfloat16)
        vae.to(dtype=torch.float16)
        image_encoder.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        text_encoder_2.to(dtype=torch.float16)

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        image_encoder.requires_grad_(False)
        transformer.requires_grad_(False)

        if not high_vram:
            # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
            DynamicSwapInstaller.install_model(transformer, device=gpu)
            DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        else:
            text_encoder.to(gpu)
            text_encoder_2.to(gpu)
            image_encoder.to(gpu)
            vae.to(gpu)
            transformer.to(gpu)

    @torch.no_grad()
    def worker(
        self,
        input_image,
        prompt,
        n_prompt,
        seed,
        total_second_length,
        latent_window_size,
        steps,
        cfg,
        gs,
        rs,
        gpu_memory_preservation,
        use_teacache,
        mp4_crf,
    ) -> str | None:
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        job_id = generate_timestamp()

        self.load_model()

        log_generate("Generating video frames...")

        high_vram = self.high_vram
        text_encoder = self.resources.get("text_encoder")
        text_encoder_2 = self.resources.get("text_encoder_2")
        tokenizer = self.resources.get("tokenizer")
        tokenizer_2 = self.resources.get("tokenizer_2")
        vae = self.resources.get("vae")
        image_encoder = self.resources.get("image_encoder")
        feature_extractor = self.resources.get("feature_extractor")
        transformer = self.resources.get("transformer")

        try:
            # Clean GPU
            if not high_vram:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

            # Text encoding

            # stream.output_queue.push(
            #     ("progress", (None, "", make_progress_bar_html(0, "Text encoding ...")))
            # )

            if not high_vram:
                fake_diffusers_current_device(
                    text_encoder, gpu
                )  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
                load_model_as_complete(text_encoder_2, target_device=gpu)

            llama_vec, clip_l_pooler = encode_prompt_conds(
                prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )

            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = (
                    torch.zeros_like(llama_vec),
                    torch.zeros_like(clip_l_pooler),
                )
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                    n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
                )

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(
                llama_vec, length=512
            )
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
                llama_vec_n, length=512
            )

            # Processing input image

            # stream.output_queue.push(
            #     (
            #         "progress",
            #         (None, "", make_progress_bar_html(0, "Image processing ...")),
            #     )
            # )

            cancel_job = False

            log_generate("Processing input image...")

            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(
                input_image, target_width=width, target_height=height
            )

            Image.fromarray(input_image_np).save(CACHE_PATH, f"{job_id}.png")

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding

            # stream.output_queue.push(
            #     ("progress", (None, "", make_progress_bar_html(0, "VAE encoding ...")))
            # )
            log_generate("VAE encoding...")

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            start_latent = vae_encode(input_image_pt, vae)

            # CLIP Vision

            # stream.output_queue.push(
            #     (
            #         "progress",
            #         (None, "", make_progress_bar_html(0, "CLIP Vision encoding ...")),
            #     )
            # )
            log_generate("CLIP Vision encoding...")

            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode(
                input_image_np, feature_extractor, image_encoder
            )
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Dtype

            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(
                transformer.dtype
            )

            # Sampling

            # stream.output_queue.push(
            #     (
            #         "progress",
            #         (None, "", make_progress_bar_html(0, "Start sampling ...")),
            #     )
            # )
            log_generate("Sampling...")

            rnd = torch.Generator("cpu").manual_seed(seed)

            history_latents = torch.zeros(
                size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32
            ).cpu()
            history_pixels = None

            history_latents = torch.cat(
                [history_latents, start_latent.to(history_latents)], dim=2
            )
            total_generated_latent_frames = 1

            for section_index in range(total_latent_sections):
                if cancel_job:
                    return

                print(
                    f"section_index = {section_index}, total_latent_sections = {total_latent_sections}"
                )

                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(
                        transformer,
                        target_device=gpu,
                        preserved_memory_gb=gpu_memory_preservation,
                    )

                if use_teacache:
                    transformer.initialize_teacache(
                        enable_teacache=True, num_steps=steps
                    )
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    preview = d["denoised"]
                    preview = vae_decode_fake(preview)

                    preview = (
                        (preview * 255.0)
                        .detach()
                        .cpu()
                        .numpy()
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    preview = einops.rearrange(preview, "b c t h w -> (b h) (t w) c")

                    if cancel_job:
                        raise KeyboardInterrupt("User ends the task.")

                    current_step = d["i"] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = f"Sampling {current_step}/{steps}"
                    desc = f"Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f} seconds (FPS-30). The video is being extended now ..."
                    # stream.output_queue.push(
                    #     (
                    #         "progress",
                    #         (preview, desc, make_progress_bar_html(percentage, hint)),
                    #     )
                    # )
                    return

                indices = torch.arange(
                    0, sum([1, 16, 2, 1, latent_window_size])
                ).unsqueeze(0)
                (
                    clean_latent_indices_start,
                    clean_latent_4x_indices,
                    clean_latent_2x_indices,
                    clean_latent_1x_indices,
                    latent_indices,
                ) = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                clean_latent_indices = torch.cat(
                    [clean_latent_indices_start, clean_latent_1x_indices], dim=1
                )

                clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[
                    :, :, -sum([16, 2, 1]) :, :, :
                ].split([16, 2, 1], dim=2)
                clean_latents = torch.cat(
                    [start_latent.to(history_latents), clean_latents_1x], dim=2
                )

                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler="unipc",
                    width=width,
                    height=height,
                    frames=latent_window_size * 4 - 3,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    # shift=3.0,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat(
                    [history_latents, generated_latents.to(history_latents)], dim=2
                )

                if not high_vram:
                    offload_model_from_device_for_memory_preservation(
                        transformer, target_device=gpu, preserved_memory_gb=8
                    )
                    load_model_as_complete(vae, target_device=gpu)

                real_history_latents = history_latents[
                    :, :, -total_generated_latent_frames:, :, :
                ]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = latent_window_size * 2
                    overlapped_frames = latent_window_size * 4 - 3

                    current_pixels = vae_decode(
                        real_history_latents[:, :, -section_latent_frames:], vae
                    ).cpu()
                    history_pixels = soft_append_bcthw(
                        history_pixels, current_pixels, overlapped_frames
                    )

                if not high_vram:
                    unload_complete_models()

                output_filename = random_filename("mp4")

                save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

                print(
                    f"Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}"
                )

                return output_filename
        except Exception as e:
            logging.error(e, exc_info=True)

            if not high_vram:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

        cancel_job = True
        return None

    def generate(self, req: Img2VidFramePackRequest):
        img = get_image_from_request(req.image)
        seed = set_seed(req.seed)
        return self.worker(
            img,
            req.prompt,
            req.negative_prompt or "",
            seed,
            req.length,
            req.latent_window_size,
            req.num_inference_steps,
            1.0,
            req.guidance_scale,
            0.0,
            6.0, # gpu_memory_preservation (6.0-128.0) larger is slower
            req.use_teacache,
            16, # mp4 compression, max 100
        )


@PluginBase.router.post("/img2vid/framepack")
async def img2vid_framepack(req: Img2VidFramePackRequest):
    plugin: Img2VidFramePackPlugin = None
    try:
        plugin = await use_plugin(Img2VidFramePackPlugin)
        result = plugin.generate(req)

        if not result:
            raise HTTPException(500, "Failed to generate video")

        return FileResponse(result, media_type="video/mp4")
    except Exception as e:
        logging.error(e, exc_info=True)
        return None


@PluginBase.router.get("/img2vid/framepack")
async def img2vid_framepack_get(req: Img2VidFramePackRequest = Depends()):
    return await img2vid_framepack(req)
