import logging
import gradio as gr
from PIL import Image
from classes.requests import Txt2ImgRequest
from modules.webui import webui
from modules.plugins import release_plugin, use_plugin
from plugins.stable_diffusion import StableDiffusionPlugin
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS
from utils.gpu_utils import random_seed_number
from utils.image_utils import base64_to_image

gallery_images = []


@webui()
def add_interface(*args, **kwargs):
    async def func(
        model: str,
        scheduler: str,
        image: Image.Image,
        image_mod_strength: float,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
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
    ):
        plugin: StableDiffusionPlugin = None

        data = None

        if seed_mode == "Random":
            seed = random_seed_number()

        try:
            model_index = SD_MODELS.index(model)

            plugin = await use_plugin(StableDiffusionPlugin)

            yield gallery.value, gr.Button("Generating Image...", interactive=False), seed

            mode = "img2img" if image is not None else "txt2img"
            req = Txt2ImgRequest(
                model_index=model_index,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
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

            data = await plugin.generate(mode, req)

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
                image = base64_to_image(data["images"][0])
                yield image, gr.Button("Generate Image", interactive=True), seed

    tab = gr.Tab(
        label="Text-to-Image",
    )

    with tab:
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    SD_MODELS,
                    label="Model",
                    value=SD_MODELS[SD_DEFAULT_MODEL_INDEX],
                )
                with gr.Row():
                    image = gr.Image(
                        label="Source Image (Optional)",
                        width=256,
                        height=256,
                        type="filepath",
                    )
                    image_mod_strength = gr.Slider(
                        0, 1, 0.5, step=0.05, label="Image Modify Strength"
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
                    inpaint_faces = gr.Radio(
                        label="Detail Faces",
                        choices=["Off", "Auto", "Custom"],
                        value="Auto",
                    )
                    face_prompt = gr.Textbox("", lines=1, label="Custom Face Prompt")
                with gr.Row():
                    width = gr.Slider(256, 2048, 768, step=128, label="Width")
                    height = gr.Slider(256, 2048, 768, step=128, label="Height")
                with gr.Row():
                    refiner_checkbox = gr.Checkbox(
                        label="Use refiner (SDXL)", value=False
                    )
                    upscale_checkbox = gr.Checkbox(
                        label="Upscale with Img2Img", value=False
                    )
                    upscale_ratio = gr.Slider(
                        1, 4, 1.5, step=0.05, label="Upscale Ratio"
                    )
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
                num_inference_steps = gr.Slider(
                    1, 100, 12, step=1, label="Inference Steps"
                )
                guidance_scale = gr.Slider(0, 10, 5, step=0.1, label="Guidance Scale")
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
                submit = gr.Button("Generate Image")
                gallery = gr.Gallery(allow_preview=True, interactive=True)

                def add_to_gallery(output):
                    logging.info("Adding to gallery")
                    gallery_images.insert(0, output)
                    yield gallery_images

                def select_from_gallery(images: list):
                    print(gallery.selected_index)
                    return images[0][0]

                submit.click(
                    func,
                    inputs=[
                        model,
                        scheduler,
                        image,
                        image_mod_strength,
                        prompt,
                        negative_prompt,
                        width,
                        height,
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
                    ],
                    outputs=[gallery, submit, seed_number],
                    queue=True,
                )


add_interface()
