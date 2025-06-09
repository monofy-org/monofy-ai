import logging
from typing import Literal

import gradio as gr
from fastapi.responses import StreamingResponse
from PIL import Image

from classes.requests import Txt2ImgRequest
from modules.plugins import release_plugin, use_plugin
from modules.webui import webui
from plugins.extras.img_canny import canny_outline
from plugins.img_depth_anything import (
    DepthRequest,
    depth_estimation,
)
from plugins.stable_diffusion import StableDiffusionPlugin
from plugins.txt2img_canny import Txt2ImgCannyPlugin
from plugins.txt2img_depth import Txt2ImgDepthMidasPlugin
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS
from utils import stable_diffusion_utils
from utils.gpu_utils import random_seed_number
from utils.image_utils import (
    base64_to_image,
)

gallery_images = []


@webui()
def add_interface(*args, **kwargs):
    async def func(
        model: str,
        scheduler: str,
        image: Image.Image,
        reference_mode: Literal["Img2Img", "Outline", "Depth"],
        image_mod_strength: float,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        hidiffusion: bool,
        num_images_per_prompt: int,
        refiner_checkbox: bool,
        upscale_checkbox: bool,
        upscale_ratio: float,
        guidance_scale: float,
        num_inference_steps: int,
        inpaint_faces: str,
        face_prompt: str,
        seed_mode: str,
        seed: int,
        censor: bool,
        gallery: list,
    ):
        if reference_mode == "Img2Img":
            T = StableDiffusionPlugin
        elif reference_mode == "Outline":
            T = Txt2ImgCannyPlugin
        elif reference_mode == "Depth":
            T = Txt2ImgDepthMidasPlugin

        plugin = None
        selected_image = None
        data = None

        if seed_mode == "Random":
            seed = random_seed_number()

        try:
            model_index = SD_MODELS.index(model)

            plugin = await use_plugin(T)

            yield None, gr.Button("Generating Image...", interactive=False), seed

            req = Txt2ImgRequest(
                model_index=model_index,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_images_per_prompt=num_images_per_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                hi=hidiffusion,
                return_json=True,
                seed=seed,
                scheduler=scheduler,
                use_refiner=refiner_checkbox,
                nsfw=not censor,
            )

            if image is not None:
                req.image = image
                req.strength = image_mod_strength

            if inpaint_faces == "Auto":
                req.face_prompt = prompt
            elif inpaint_faces == "Custom":
                req.face_prompt = face_prompt

            if upscale_checkbox:
                req.upscale = upscale_ratio

            data = await plugin.generate(req)

        except Exception as e:
            logging.error(e, exc_info=True)
            yield None, gr.Button("Generate Image", interactive=True), seed
            raise gr.Error(
                "I couldn't generate this image. Please make sure the prompt doesn't violate guidelines."
            )

        finally:
            if plugin is not None:
                release_plugin(StableDiffusionPlugin)

            if data is not None:
                images = [base64_to_image(image) for image in data["images"]]
                yield (
                    images + gallery if gallery else images,
                    gr.Button("Generate Image", interactive=True),
                    seed,
                )

    tab = gr.Tab(
        label="Text-to-Image",
    )

    with tab:
        with gr.Row():
            with gr.Column(scale=0.5):
                model = gr.Dropdown(
                    SD_MODELS,
                    label="Model",
                    value=SD_MODELS[SD_DEFAULT_MODEL_INDEX],
                )
                prompt = gr.Textbox(
                    "friendly humanoid cyborg robot with cobalt plating, in space, depth of field, turning to look at the camera",
                    lines=3,
                    label="Prompt",
                )
                negative_prompt = gr.Textbox(
                    "nsfw, blurry, deformed, worst quality",
                    lines=3,
                    label="Negative Prompt",
                )
                with gr.Row():
                    width = gr.Slider(256, 2048, 512, step=128, label="Width")
                    height = gr.Slider(256, 2048, 768, step=128, label="Height")
                with gr.Row():
                    guidance_scale = gr.Slider(0, 10, 5, step=0.1, label="Guidance Scale")
                    num_inference_steps = gr.Slider(
                        1, 100, 14, step=1, label="Inference Steps"
                    )                    
                with gr.Row():
                    hidiffusion = gr.Checkbox(
                        label="HiDiffusion",
                        value=False,
                    )
                    refiner_checkbox = gr.Checkbox(label="SDXL Refiner", value=False)
                with gr.Row():
                    upscale_checkbox = gr.Checkbox(
                        label="Upscale with Img2Img", value=False
                    )
                    upscale_ratio = gr.Slider(
                        1, 4, 1.5, step=0.05, label="Upscale Ratio"
                    )                                        
                with gr.Accordion(label="Reference Image", open=False):
                    with gr.Row():
                        image = gr.Image(
                            label="Source Image (Optional)",
                            width=256,
                            height=256,
                            type="filepath",
                        )
                        features = gr.Image(
                            "Extracted Features",
                            width=256,
                            height=256,
                            interactive=False,
                            visible=False,
                        )

                    with gr.Column():
                        image_mod_strength = gr.Slider(
                            0, 1, 0.5, step=0.05, label="Image Modify Strength"
                        )
                        reference_mode = gr.Radio(
                            label="Reference Mode",
                            choices=["Img2Img", "Outline", "Depth"],
                            value="Img2Img",
                        )

                        async def extract_features(image, reference_mode):
                            yield gr.Image(visible=True)
                            if reference_mode == "Depth":
                                response: dict = await depth_estimation(
                                    DepthRequest(image=image, return_json=True)
                                )
                                base64_image = response["images"][0]
                                yield gr.Image(
                                    label="Depth",
                                    visible=True,
                                    value=base64_to_image(base64_image),
                                )
                            elif reference_mode == "Outline":
                                yield gr.Image(
                                    label="Canny Outline",
                                    visible=True,
                                    value=canny_outline(image=image),
                                )
                            else:
                                yield gr.Image(visible=False)

                        reference_mode.select(
                            extract_features,
                            inputs=[image, reference_mode],
                            outputs=[features],
                        )
                with gr.Row():
                    inpaint_faces = gr.Radio(
                        label="Detail Faces",
                        choices=["Off", "Auto", "Custom"],
                        value="Auto",
                    )
                    face_prompt = gr.Textbox("", lines=1, label="Custom Face Prompt")
                    quick_fix = gr.Button("Quick Fix", scale=0.5)                
                    
                with gr.Row():
                    seed_mode = gr.Radio(
                        ["Random", "Fixed"], value="Random", label="Seed"
                    )
                    seed_number = gr.Number(
                        -1,
                        maximum=2**64 - 1,
                        minimum=-1,
                        precision=0,
                        label="Seed Number",
                    )
                scheduler = gr.Dropdown(
                    [
                        "ddim",
                        "dpm2m",
                        "euler_a",
                        "euler",
                        "heun",
                        "lms",
                        "sde",
                        "tcd",
                    ],
                    value="sde",
                    label="Scheduler",
                )
                censor = gr.Checkbox(label="Censor NSFW", value=True)
            with gr.Column():
                with gr.Row(scale=0.5):
                    with gr.Row():
                        num_images_per_prompt = gr.Slider(
                            1, 8, 1, step=1, label="Images per prompt"
                        )
                        generate_button = gr.Button("Generate Image")
                gallery = gr.Gallery(
                    format="png",
                    interactive=False,
                    allow_preview=False,
                    height=140,
                    rows=[1],
                    columns=[8],
                    object_fit="contain",
                )
                selected_image = gr.Image(format="png")

                def gallery_select(selection: gr.SelectData):
                    image_info = selection.value.get("image")
                    return image_info.get("path") if image_info else None

                gallery.select(gallery_select, inputs=None, outputs=[selected_image])

                def update_selected_image(images):
                    if images and images[0] and images[0][0]:
                        yield images[0][0]
                    else:
                        raise gr.Error("No images were generated.")

                generate_button.click(
                    func,
                    inputs=[
                        model,
                        scheduler,
                        image,
                        reference_mode,
                        image_mod_strength,
                        prompt,
                        negative_prompt,
                        width,
                        height,
                        hidiffusion,
                        num_images_per_prompt,
                        refiner_checkbox,
                        upscale_checkbox,
                        upscale_ratio,
                        guidance_scale,
                        num_inference_steps,
                        inpaint_faces,
                        face_prompt,
                        seed_mode,
                        seed_number,
                        censor,
                        gallery,
                    ],
                    outputs=[gallery, generate_button, seed_number],
                    queue=True,
                ).then(
                    update_selected_image, inputs=[gallery], outputs=[selected_image]
                )

                async def quick_fix_callback(
                    image, face_prompt, prompt, model, gallery
                ):
                    plugin: StableDiffusionPlugin = None
                    try:
                        input_image = Image.fromarray(image)
                        model_index = SD_MODELS.index(model)

                        plugin = await use_plugin(StableDiffusionPlugin)
                        req = Txt2ImgRequest(
                            prompt=face_prompt or prompt,
                            steps=0,
                            face_prompt=face_prompt,
                            model_index=model_index,
                        )
                        plugin.load_model(model_index)
                        inpaint = plugin.resources.get("inpaint")
                        if not inpaint:
                            raise gr.Error("Inpaint model not found.")
                        result = stable_diffusion_utils.inpaint_faces(
                            inpaint, input_image, req
                        )
                        yield [result] + gallery if gallery else [result]
                    except Exception as e:
                        raise gr.Error(e)
                    finally:
                        if plugin:
                            release_plugin(plugin)

                quick_fix.click(
                    quick_fix_callback,
                    inputs=[selected_image, face_prompt, prompt, model, gallery],
                    outputs=[gallery],
                )


add_interface()
