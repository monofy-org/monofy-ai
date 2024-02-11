import gc
import logging
import numpy as np
import os
import torch
from cv2 import Canny
from PIL import ImageFilter
from settings import (
    SD_DEFAULT_MODEL_INDEX,
    SD_DEFAULT_STEPS,
    SD_MODELS,
    SD_USE_SDXL,
    SD_USE_VAE,
    SD_CLIP_SKIP,
    SD_DEFAULT_SCHEDULER,
    SD_COMPILE_UNET,
    SD_COMPILE_VAE,
    USE_DEEPSPEED,
    USE_XFORMERS,
)
from utils.file_utils import import_model
from utils.gpu_utils import (
    autodetect_device,
    autodetect_dtype,
    set_seed,
)
from PIL import Image
from diffusers.schedulers import EulerAncestralDiscreteScheduler
from diffusers import (
    SchedulerMixin,
    DiffusionPipeline,
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    DPMSolverSDEScheduler,
    # LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    DDIMScheduler,
    StableVideoDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
from utils.image_utils import create_upscale_mask
from huggingface_hub import hf_hub_download
from submodules.frame_interpolation.eval import interpolator
film_interpolator = interpolator.Interpolator("models/film_net/Style/saved_model", None)

# from insightface.app import FaceAnalysis
# from ip_adapter.ip_adapter_faceid import IPAdapterFaceID


friendly_name = "diffusers"
logging.warn(f"Initializing {friendly_name}...")
device = autodetect_device()
pipelines: dict[DiffusionPipeline] = {}
controlnets: dict[ControlNetModel] = {}
vae: AutoencoderKL = None
text_encoder: CLIPTextModel = None
text_encoder_2: CLIPTextModelWithProjection = None
schedulers: dict[SchedulerMixin] = {}
image_pipeline: StableDiffusionXLPipeline | StableDiffusionPipeline = None
current_model = None
default_steps = SD_DEFAULT_STEPS


def load_model(repo_or_path: str = SD_MODELS[SD_DEFAULT_MODEL_INDEX]):
    global vae
    global text_encoder
    global text_encoder_2
    global image_pipeline
    global current_model
    global default_steps

    if SD_USE_VAE and not vae:
        vae = import_model(AutoencoderKL, "stabilityai/sd-vae-ft-mse")

    if not text_encoder:
        CLIP_MODEL = "openai/clip-vit-large-patch14"
        clip_config = CLIPTextConfig.from_pretrained(
            CLIP_MODEL,
            #        local_dir=os.path.join("models", CLIP_MODEL),
            #        local_dir_use_symlinks=False,
            device=device,            
        )
        clip_config.num_hidden_layers = 12 - SD_CLIP_SKIP

        text_encoder = CLIPTextModel(clip_config)
        text_encoder.to(device=device, dtype=autodetect_dtype(bf16_allowed=False))

    # if not text_encoder_2:
    #    CLIP_MODEL_2 = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    #    clip_config_2 = CLIPTextConfig.from_pretrained(
    #        CLIP_MODEL_2,
    #        local_dir=os.path.join("models", CLIP_MODEL_2),
    #        local_dir_use_symlinks=False,
    #    )
    #    text_encoder_2 = CLIPTextModelWithProjection(clip_config_2)
    #    text_encoder_2.to(device=device, dtype=autodetect_dtype())

    if repo_or_path != current_model or not image_pipeline:

        if image_pipeline:
            image_pipeline.maybe_free_model_hooks()
            del image_pipeline

        image_pipeline_type = (
            StableDiffusionXLPipeline if SD_USE_SDXL else StableDiffusionPipeline
        )

        single_file = repo_or_path.endswith(".safetensors")

        if os.path.exists(repo_or_path):
            model_path = repo_or_path

        elif single_file:
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

        else:
            model_path = repo_or_path

        image_pipeline = import_model(
            image_pipeline_type,
            model_path,
            device=autodetect_device(),
        )

        current_model = repo_or_path

        image_pipeline.enable_vae_tiling()
        image_pipeline.unet.eval()
        image_pipeline.vae.eval()
        # image_pipeline.vae.force_upscale = True
        # image_pipeline.vae.use_tiling = True
        image_pipeline.scheduler.config["lower_order_final"] = not SD_USE_SDXL
        image_pipeline.scheduler.config["use_karras_sigmas"] = True
        
        turbo = "turbo" in current_model.lower()
        default_steps = 15 if turbo else 18 if SD_USE_SDXL else 25

        init_schedulers()

        dtype = autodetect_dtype()

        pipelines["txt2img"] = AutoPipelineForText2Image.from_pipe(
            image_pipeline,
            device=device,
            dtype=dtype,
            scheduler=schedulers[SD_DEFAULT_SCHEDULER],
        )

        pipelines["img2img"] = AutoPipelineForImage2Image.from_pipe(
            image_pipeline,
            device=device,
            dtype=dtype,
            scheduler=schedulers[SD_DEFAULT_SCHEDULER],
        )

        pipelines["inpaint"] = AutoPipelineForInpainting.from_pipe(
            image_pipeline,
            device=device,
            dtype=dtype,
            scheduler=schedulers[SD_DEFAULT_SCHEDULER],
        )

        # compile model (linux only)
        if not os.name == "nt":
            if SD_COMPILE_UNET:
                image_pipeline.unet = torch.compile(image_pipeline.unet).to(device)
            if SD_COMPILE_VAE:
                image_pipeline.vae = torch.compile(image_pipeline.vae).to(device)

        if USE_XFORMERS:
            from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

            if not USE_DEEPSPEED:
                image_pipeline.enable_xformers_memory_efficient_attention(
                    attention_op=MemoryEfficientAttentionFlashAttentionOp
                )
                image_pipeline.vae.enable_xformers_memory_efficient_attention(
                    attention_op=None  # skip attention op for VAE
                )


# Currently not used
# preview_vae = AutoencoderTiny.from_pretrained(
#    "madebyollin/taesd",
#    # variant="fp16" if USE_FP16 else None, # no fp16 available
#    torch_dtype=dtype,
#    safetensors=True,
#    device=device,
#    cache_dir=os.path.join("models", "VAE"),
# )

# controlnets["canny"] = import_model(
#    ControlNetModel, "lllyasviel/sd-controlnet-canny", set_variant_fp16=False
# )
# controlnets["depth"] = import_model(
#    ControlNetModel, "lllyasviel/sd-controlnet-depth", set_variant_fp16=False
# )


def init_img2vid():
    global pipelines

    if "img2vid" not in pipelines:
        model_name = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        path = os.path.join(
            "models", model_name
        )
        pipelines["img2vid"] = StableVideoDiffusionPipeline.from_pretrained(
            path if os.path.exists(path) else model_name,
            use_safetensors=True,                        
            device=device,
            torch_dtype=autodetect_dtype(bf16_allowed=False),
            variant="fp16",
            local_dir=path,
            local_dir_use_symlinks=False,            
        )
        pipelines["img2vid"].enable_sequential_cpu_offload()
        pipelines["img2vid"].to(dtype=autodetect_dtype(bf16_allowed=False))
        pipelines["img2vid"].enable_xformers_memory_efficient_attention()
        pipelines["img2vid"].vae.to(dtype=autodetect_dtype(bf16_allowed=False))
        pipelines["img2vid"].vae.enable_xformers_memory_efficient_attention()


def init_txt2vid():
    pipelines["txt2vid"] = import_model(
        DiffusionPipeline, "cerspense/zeroscope_v2_576w", device=device
    )
    pipelines["txt2vid"].to(device=device, dtype=autodetect_dtype())
    if USE_XFORMERS:
        pipelines["txt2vid"].enable_xformers_memory_efficient_attention()
        pipelines["txt2vid"].vae.enable_xformers_memory_efficient_attention()


def init_schedulers():
    global schedulers

    schedulers["euler"] = EulerDiscreteScheduler.from_config(
        image_pipeline.scheduler.config
    )
    schedulers["euler_a"] = EulerAncestralDiscreteScheduler.from_config(
        image_pipeline.scheduler.config
    )
    schedulers["sde"] = DPMSolverSDEScheduler.from_config(
        image_pipeline.scheduler.config
    )
    # schedulers["lms"]: LMSDiscreteScheduler = LMSDiscreteScheduler.from_config(
    #    image_pipeline.scheduler.config
    # )
    schedulers["heun"] = HeunDiscreteScheduler.from_config(
        image_pipeline.scheduler.config
    )
    schedulers["ddim"] = DDIMScheduler.from_config(image_pipeline.scheduler.config)


def create_controlnet_pipeline(name: str):
    pipelines[name] = StableDiffusionControlNetImg2ImgPipeline(
        vae=image_pipeline.vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=image_pipeline.tokenizer,
        unet=image_pipeline.unet,
        controlnet=controlnets[name],
        scheduler=image_pipeline.scheduler,
        safety_checker=None,
        # image_processor=image_pipeline.image_processor,
        feature_extractor=image_pipeline.image_processor,
        requires_safety_checker=False,
    )


# create_controlnet_pipeline("canny")
# create_controlnet_pipeline("depth")


def widen(
    image,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    aspect_ratio: float,
    seed: int = -1,
):
    mask_image = create_upscale_mask(width, height, aspect_ratio)
    set_seed(seed)
    return pipelines["inpaint"](
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        mask_image=mask_image,
    )


def upscale(
    image,
    original_width: int,
    original_height: int,
    prompt: str,
    negative_prompt: str,
    steps: int,
    strength: float = 0.6,
    controlnet: str = None,  # "canny" or "depth
    upscale_coef: float = 0,
    seed: int = -1,
):
    if steps > 100:
        logging.warn(f"Limiting steps to 100 from {steps}")
        steps = 100

    if strength > 2:
        logging.warn(f"Limiting strength to 2 from {strength}")
        strength = 2
    upscaled_image = image.resize(
        (int(original_width * upscale_coef), int(original_height * upscale_coef)),
        Image.Resampling.BICUBIC,
    )

    set_seed(seed)

    if controlnet == "canny":
        np_img = np.array(image)
        outline = Canny(np_img, 100, 200)
        outline = outline[:, :, None]
        outline = np.concatenate([outline, outline, outline], axis=2)

        # DEBUG
        canny_image = Image.fromarray(outline)
        canny_image.save("canny.png")

        return pipelines["canny"](
            prompt=prompt,
            image=outline,
            control_image=image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            strength=strength,
            guidance_scale=strength * 10,
        ).images[0]
    else:
        upscaled_image = pipelines["img2img"](
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=upscaled_image,
            num_inference_steps=steps,
            strength=strength,
        ).images[0]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return upscaled_image


def offload(for_task: str):
    global image_pipeline

    # logging.info(f"Switching to {for_task}...")
    if for_task == "txt2vid":
        image_pipeline.maybe_free_model_hooks()
        if "img2vid" in pipelines:
            pipelines["img2vid"].maybe_free_model_hooks()
    if for_task == "img2vid":
        image_pipeline.maybe_free_model_hooks()
        if "txt2vid" in pipelines:
            pipelines["txt2vid"].maybe_free_model_hooks()
    elif for_task == "sdxl" or for_task == "stable diffusion":
        if "img2vid" in pipelines:
            pipelines["img2vid"].maybe_free_model_hooks()
        if "txt2vid" in pipelines:
            pipelines["txt2vid"].maybe_free_model_hooks()


def fix_faces(
    image: Image.Image, seed: int = -1, face_prompt: str = None, **img2img_kwargs
):
    from submodules.adetailer.adetailer.mediapipe import mediapipe_face_mesh

    # DEBUG
    # image.save("face-fix-before.png")
    # convert image to black and white

    if face_prompt is None:
        face_prompt = img2img_kwargs.get("prompt", "")
    else:
        img2img_kwargs["prompt"] = face_prompt

    black_and_white = image.convert("L").convert("RGB")

    output = mediapipe_face_mesh(black_and_white, confidence=0.1)
    faces_count = len(output.bboxes)

    if faces_count == 0:
        logging.info("No faces found")
        return image

    logging.info(f"Fixing {faces_count} face{ 's' if faces_count > 1 else '' }...")

    # find the biggest face
    # biggest_face = 0
    biggest_face_size = 0
    for i in range(faces_count):
        bbox = output.bboxes[i]
        size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if size > biggest_face_size:
            biggest_face_size = size
            # biggest_face = i

    # convert bboxes to squares
    for i in range(faces_count):
        bbox = output.bboxes[i]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        diff = abs(width - height)
        if width < height:
            bbox[0] = bbox[0] - diff // 2
            bbox[2] = bbox[2] + diff // 2
        else:
            bbox[1] = bbox[1] - diff // 2
            bbox[3] = bbox[3] + diff // 2
        output.bboxes[i] = bbox

    # Extends boxes in each direction by pixel_buffer.
    # Provides additional context at the cost of quality.
    face_context_buffer = 32

    for i in range(faces_count):
        bbox = output.bboxes[i]
        bbox[0] = bbox[0] - face_context_buffer
        bbox[1] = bbox[1] - face_context_buffer
        bbox[2] = bbox[2] + face_context_buffer
        bbox[3] = bbox[3] + face_context_buffer
        output.bboxes[i] = bbox

    face_mask_blur = 0.05 * max(bbox[2] - bbox[0], bbox[3] - bbox[1])

    for i in range(faces_count):
        # skip if less than 10% of the image size
        if (output.bboxes[i][2] - output.bboxes[i][0]) * (
            output.bboxes[i][3] - output.bboxes[i][1]
        ) < (biggest_face_size * 0.8):
            logging.info(f"Skipping face #{i+1} (background)")
            continue

        mask = output.masks[i]
        face = image.crop(output.bboxes[i])
        face_mask = mask.crop(output.bboxes[i])
        bbox = output.bboxes[i]

        # DEBUG
        # if i == biggest_face:
        #    face.save("face-image.png")
        #    face_mask.save("face-mask.png")

        set_seed(seed)
        image2 = pipelines["inpaint"](
            image=face, mask_image=face_mask, **img2img_kwargs
        ).images[0]

        face_mask = face_mask.filter(ImageFilter.GaussianBlur(face_mask_blur))

        # DEBUG
        # if i == biggest_face:
        #    image2.save("face-image2.png")

        image2 = image2.resize((bbox[2] - bbox[0], bbox[3] - bbox[1]))

        # DEBUG
        # if i == biggest_face:
        #    image2.save("face-image2-small.png")

        image.paste(image2, (bbox[0], bbox[1]), mask=face_mask)

    # DEBUG
    # image.save("face-fix-after.png")

    return image
