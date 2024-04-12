import logging
import gradio as gr
from classes.requests import Txt2ImgRequest
from modules.webui import webui
from modules.plugins import release_plugin, use_plugin
from plugins.stable_diffusion import StableDiffusionPlugin
from utils.stable_diffusion_utils import postprocess


@webui(section="Txt2Img")
def add_interface(*args, **kwargs):
    async def func(
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        guidance_scale: float,
        num_inference_steps: int,
    ):
        plugin: StableDiffusionPlugin = None

        try:
            plugin = await use_plugin(StableDiffusionPlugin)

            yield output, gr.Button("Generating Image...", interactive=False)

            mode = "txt2img"
            req = Txt2ImgRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                return_json=False,
            )
            image = await plugin.generate(mode, req)

            image, json_response = await postprocess(
                plugin,
                image,
                req,
            )
        except Exception as e:
            logging.error(e, exc_info=True)
            raise e
        finally:
            if plugin is not None:
                release_plugin(StableDiffusionPlugin)

            yield image, gr.Button("Generate Image", interactive=True)
        

    tab = gr.Tab(
        label="Text-to-Image",
    )

    with tab:
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox("humanoid cyborg robot, dark factory, depth of field", lines=3, label="Prompt")
                negative_prompt = gr.Textbox("blurry, deformed, worst quality", lines=3, label="Negative Prompt")
                width = gr.Slider(256, 2048, 768, step=128, label="Width")
                height = gr.Slider(256, 2048, 768, step=128, label="Height")
                guidance_scale = gr.Slider(0, 10, 2, step=0.1, label="Guidance Scale")
                num_inference_steps = gr.Slider(1, 100, 8, step=1, label="Inference Steps")
            with gr.Column():
                output = gr.Image(label="Output")
                submit = gr.Button("Generate Image")

                submit.click(
                    func,
                    inputs=[
                        prompt,
                        negative_prompt,
                        width,
                        height,
                        guidance_scale,
                        num_inference_steps,
                    ],
                    outputs=[output, submit],
                )


add_interface()
