import logging
import gradio as gr
from classes.requests import Txt2ImgRequest
from modules.webui import webui
from modules.plugins import release_plugin, use_plugin
from plugins.stable_diffusion import StableDiffusionPlugin
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS
from utils.gpu_utils import random_seed_number
from utils.image_utils import base64_to_image


@webui(section="Txt2Img")
def add_interface(*args, **kwargs):
    async def func(
        model: str,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        guidance_scale: float,
        num_inference_steps: int,
        inpaint_faces: str,
        face_prompt: str,
        seed_mode: str,
        seed: int,
        censor: bool,
    ):
        plugin: StableDiffusionPlugin = None
        image = None

        if seed_mode == "Random":
            seed = random_seed_number()

        try:
            model_index = SD_MODELS.index(model)

            plugin = await use_plugin(StableDiffusionPlugin)

            yield output, gr.Button("Generating Image...", interactive=False), seed

            mode = "txt2img"
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
                nsfw=not censor,
            )

            if inpaint_faces == "Auto":
                req.face_prompt = prompt
            elif inpaint_faces == "Custom":
                req.face_prompt = face_prompt

            data = await plugin.generate(mode, req)            

        except Exception as e:
            logging.error(e, exc_info=True)
            raise e
        finally:
            if plugin is not None:
                release_plugin(StableDiffusionPlugin)            
            
            image = base64_to_image(data["images"][0])

            yield image, gr.Button("Generate Image", interactive=True), seed

    tab = gr.Tab(
        label="Text-to-Image",
    )

    with tab:
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    SD_MODELS, label="Model", value=SD_MODELS[SD_DEFAULT_MODEL_INDEX]
                )
                prompt = gr.Textbox(
                    "humanoid cyborg robot, dark factory, depth of field, turning to look at the camera",
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
                with gr.Row():
                    width = gr.Slider(256, 2048, 768, step=128, label="Width")
                    height = gr.Slider(256, 2048, 768, step=128, label="Height")
                num_inference_steps = gr.Slider(
                    1, 100, 8, step=1, label="Inference Steps"
                )
                guidance_scale = gr.Slider(0, 10, 2, step=0.1, label="Guidance Scale")
                censor = gr.Checkbox(label="Censor NSFW", value=True)
            with gr.Column():
                output = gr.Image(label="Output")
                submit = gr.Button("Generate Image")

                submit.click(
                    func,
                    inputs=[
                        model,
                        prompt,
                        negative_prompt,
                        width,
                        height,
                        guidance_scale,
                        num_inference_steps,
                        inpaint_faces,
                        face_prompt,
                        seed_mode,
                        seed_number,
                        censor,
                    ],
                    outputs=[output, submit, seed_number],
                )


add_interface()
