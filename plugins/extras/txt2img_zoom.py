import logging
from typing import Optional
from fastapi import BackgroundTasks, HTTPException
from pydantic import BaseModel
from classes.requests import Txt2ImgRequest
from modules.filter import filter_request
from modules.plugins import release_plugin, router, use_plugin
from plugins.img_rembg import RembgPlugin
from plugins.stable_diffusion import StableDiffusionPlugin, format_response
from plugins.video_plugin import VideoPlugin
from utils.gpu_utils import random_seed_number, set_seed
from utils.image_utils import (
    extend_image,
    get_image_from_request,
    image_to_base64_no_header,
)
from PIL import Image, ImageFilter, ImageOps

from utils.stable_diffusion_utils import postprocess
from diffusers.utils import make_image_grid

from utils.text_utils import generate_combinations


class Txt2ImgZoomRequest(BaseModel):
    image: str
    prompt: str
    negative_prompt: Optional[str] = None
    face_prompt: Optional[str] = None
    strength: float = 0.75
    guidance_scale: float = 6.5
    num_inference_steps: int = 16
    width: int = 768
    height: int = 768
    model_index: int = 0
    upscale: Optional[float] = 0
    nsfw: bool = False
    video: bool = False
    return_json: bool = False
    seed: int = -1
    image_grid: bool = False
    include_all_images: bool = False
    include_steps: bool = True
    repeat: int = 1
    # position: Literal["left", "right", "top", "bottom", "center"] = "center"


class Txt2ImgZoomPlugin(VideoPlugin):

    name = "Stable Diffusion"
    description = "Base plugin for txt2img, img2img, inpaint, etc."
    instance = None
    plugins = ["StableDiffusionPlugin", "RembgPlugin"]

    def __init__(self):
        super().__init__()
        self._sd: StableDiffusionPlugin = None
        self._rembg: RembgPlugin = None

    async def get_sd(self):
        if self._sd is None:
            self._sd = await use_plugin(StableDiffusionPlugin, True)
        return self._sd

    async def get_rembg(self):
        if self._rembg is None:
            self._rembg = await use_plugin(RembgPlugin, True)
        return self._rembg


@router.post("/txt2img/zoom", tags=["Image Generation"])
async def txt2img_zoom(
    background_tasks: BackgroundTasks,
    req: Txt2ImgZoomRequest,
):
    scale = 0.75
    mask_border = 64
    inpaint_border = 96

    image = get_image_from_request(req.image, (req.width, req.height))
    prompts = generate_combinations(req.prompt)
    neg_prompts = generate_combinations(req.prompt)
    req.width = image.width
    req.height = image.height

    if len(prompts) > 1:
        req.repeat = len(prompts)
        if len(prompts) > 3:
            raise HTTPException(
                status_code=400,
                detail="Too many prompts. Maximum 3 prompts allowed.",
            )

    req = filter_request(req)

    if req.seed == -1:
        req.seed = random_seed_number()

    num_inference_steps = req.num_inference_steps

    images = []
    frames = []
    images.append(image)
    frames.append(image)

    if scale == 1:
        pass

    else:
        plugin: Txt2ImgZoomPlugin = None

        try:
            plugin = await use_plugin(Txt2ImgZoomPlugin)
            sd_plugin = await plugin.get_sd()
            rembg_plugin = await plugin.get_rembg()

            for i in range(req.repeat):

                prompt = prompts[min(i, len(prompts) - 1)]
                neg_prompt = neg_prompts[min(i, len(neg_prompts) - 1)]

                logging.info(f"Generating image {i + 1}/{req.repeat}: {prompt}")

                if req.include_steps and i > 0:
                    images.append(image)

                new_size = (int(image.width * scale), int(image.height * scale))

                if scale > 1:
                    # zoom and crop (centered)
                    cropped_image = image.resize(new_size)
                    left = (image.width - req.width) // 2
                    top = (image.height - req.height) // 2
                    right = (image.width + req.width) // 2
                    bottom = (image.height + req.height) // 2
                    expanded_image = cropped_image.crop((left, top, right, bottom))

                else:

                    left = (req.width - new_size[0]) // 2
                    top = (req.height - new_size[1]) // 2

                    border_h = 256
                    border_v = 256

                    expanded_image, inpaint_mask = extend_image(
                        image,
                        h=border_h,
                        v=border_v,
                        with_mask=True,
                        mask_border=mask_border,
                    )
                    inpaint_mask = inpaint_mask.filter(
                        ImageFilter.GaussianBlur(mask_border)
                    )
                    inpaint_mask = inpaint_mask.point(lambda p: 0 if p < 128 else p)

                    if req.include_all_images:
                        images.append(expanded_image)
                        images.append(inpaint_mask)

                    sd_plugin.load_model(req.model_index)

                    inpaint = sd_plugin.resources["inpaint"]
                    inpaint_kwargs = {
                        "prompt": prompt,
                        "negative_prompt": neg_prompt,
                        "image": expanded_image,
                        "mask_image": inpaint_mask,
                        "num_inference_steps": num_inference_steps,
                        "guidance_scale": 2,
                        "strength": req.strength,
                        "width": req.width,
                        "height": req.height,                        
                    }

                    # req.seed = set_seed(req.seed)

                    inpainted_image: Image.Image = inpaint(**inpaint_kwargs).images[0]
                    if req.include_all_images or req.include_steps:
                        images.append(inpainted_image)

                    postprocess_req = Txt2ImgRequest(
                        prompt=prompts[i] if i < len(prompts) else req.prompt,
                        negative_prompt=(
                            neg_prompts[i]
                            if i < len(neg_prompts)
                            else req.negative_prompt
                        ),
                        width=req.width,
                        height=req.height,
                        guidance_scale=3,
                        strength=0.5,
                        num_inference_steps=num_inference_steps,
                        seed=req.seed,
                        upscale=req.upscale,
                        nsfw=req.nsfw,
                    )

                    postprocessed_image, json_response = await postprocess(
                        sd_plugin, inpainted_image, postprocess_req
                    )
                    if req.include_all_images or req.include_steps:
                        images.append(postprocessed_image)

                    if req.upscale:

                        expanded_mask = ImageOps.expand(
                            inpaint_mask,
                            border=mask_border + inpaint_border,
                            fill="white",
                        )
                        expanded_mask = expanded_mask.filter(
                            ImageFilter.GaussianBlur(mask_border)
                        )

                        small_image = expanded_image.resize(postprocessed_image.size)
                        small_mask = expanded_mask.resize(postprocessed_image.size)
                        small_mask = ImageOps.invert(small_mask)

                        if req.include_all_images:
                            images.append(small_image)
                            images.append(small_mask)

                        final_image = postprocessed_image.copy()
                        final_image.paste(
                            small_image, (0, 0), mask=small_mask.convert("L")
                        )
                        final_image = final_image.resize((req.width, req.height))
                        images.append(final_image)

                        image = final_image.copy()

                        frames.append(final_image)

                if req.video:
                    return plugin.video_response(background_tasks, frames, 12, 5, 1)

                elif not req.image_grid:
                    json_response["images"] = [image_to_base64_no_header(image)]

        except Exception as e:
            logging.error(e, exc_info=True)
            return {"error": str(e)}

        finally:
            if plugin is not None:
                release_plugin(Txt2ImgZoomPlugin)

        if req.image_grid:
            cols = min(4, len(images))
            while len(images) % cols != 0:
                images.append(Image.new("RGBA", (req.width, req.height), (0, 0, 0, 0)))
            grid = make_image_grid(images, len(images) // cols, cols, 512)
            json_response["images"] = [image_to_base64_no_header(grid)]

        return format_response(req, json_response)
