import logging
from typing import Optional
from pydantic import BaseModel
from modules.filter import filter_request
from modules.plugins import release_plugin, router, use_plugin
from plugins.stable_diffusion import StableDiffusionPlugin, format_response
from utils.image_utils import (
    extend_image,
    get_image_from_request,
    image_to_base64_no_header,
)
from PIL import Image, ImageFilter, ImageOps

from utils.stable_diffusion_utils import postprocess
from diffusers.utils import make_image_grid

class Txt2ImgZoomRequest(BaseModel):
    image: str
    prompt: str
    negative_prompt: Optional[str] = None
    face_prompt: Optional[str] = None
    strength: float = 1.0
    guidance_scale: float = 1.0
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


@router.post("/txt2img/zoom", tags=["Image Generation"])
async def txt2img_zoom(
    req: Txt2ImgZoomRequest,
):
    image = get_image_from_request(req.image)
    req.width = image.width
    req.height = image.height

    req = filter_request(req)

    images = []
    images.append(image)

    # the point of this function is to take the original image and either zoom in or zoom out depending on the strength
    # it should zoom on the center of the image
    # the strength should be a float between 0.25 and 4
    # if the strength is 1, the image should not be modified
    # if the strength is greater than 1, the image should be zoomed in and cropped
    # if the strength is less than 1, the image should be zoomed out and padded with black

    scale = 0.75

    if scale == 1:
        pass

    else:
        new_size = (int(image.width * scale), int(image.height * scale))

        if scale > 1:
            # zoom and crop (centered)
            cropped_image = image.resize(new_size)
            left = (image.width - req.width) // 2
            top = (image.height - req.height) // 2
            right = (image.width + req.width) // 2
            bottom = (image.height + req.height) // 2
            noise_image = cropped_image.crop((left, top, right, bottom))

        else:

            left = (req.width - new_size[0]) // 2
            top = (req.height - new_size[1]) // 2

            mask_border = 64

            border_h = 256
            border_v = 256

            noise_image, mask = extend_image(
                image,
                h=border_h,
                v=border_v,
                with_mask=True,
                mask_border=mask_border,
            )
            images.append(noise_image.copy())
            images.append(mask.copy())

        plugin = None
        try:
            plugin: StableDiffusionPlugin = await use_plugin(StableDiffusionPlugin)
            plugin._load_model(req.model_index)
            inpaint = plugin.resources["inpaint"]
            kwargs = {
                "prompt": req.prompt,
                "image": noise_image,
                "mask_image": mask,
                "num_inference_steps": req.num_inference_steps or 16,
                "strength": req.strength,
                "width": req.width,
                "height": req.height,
            }

            new_image: Image.Image = inpaint(**kwargs).images[0]
            images.append(new_image.copy())

            mask = mask.filter(ImageFilter.GaussianBlur(mask_border))
            # images.append(mask.copy())

            # noise_image.paste(new_image, (0, 0), mask.convert("L"))            
            # images.append(noise_image.copy())

            req.num_inference_steps = 18
            req.strength = 0.4
            req.guidance_scale = 6.5

            new_image, json_response = await postprocess(plugin, new_image, req)
            images.append(new_image.copy())

            if req.upscale:

                expanded_image = ImageOps.expand(
                    image, border=(border_h, border_v, border_h, border_v), fill="black"
                )
                expanded_mask = ImageOps.expand(
                    mask, border=(border_h, border_v, border_h, border_v), fill="white"
                )

                small_image = expanded_image.resize((new_image.width, new_image.height))
                small_mask = expanded_mask.resize((new_image.width, new_image.height))
                small_mask = ImageOps.invert(small_mask)

                images.append(small_image.copy())
                images.append(small_mask.copy())

                new_image.paste(small_image, (0, 0), small_mask.convert("L"))
                new_image = new_image.resize((req.width, req.height))
                images.append(new_image.copy())

                # add a blank image
                # images.append(Image.new("RGBA", (req.width, req.height), (0, 0, 0, 0)))

                if req.image_grid:
                    new_image = make_image_grid(images, 2, 4, resize=256)

                json_response["images"] = [image_to_base64_no_header(new_image)]

            return format_response(req, json_response)

        except Exception as e:
            logging.error(e, exc_info=True)

        finally:
            if plugin is not None:
                release_plugin(StableDiffusionPlugin)
