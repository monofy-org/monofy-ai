import gradio as gr
from modules.webui import webui
from classes.requests import Txt2VidRequest
from modules.plugins import release_plugin, use_plugin
from plugins.txt2vid import Txt2VidZeroPlugin
from plugins.txt2vid_animate import Txt2VidAnimatePlugin
from plugins.txt2vid_zeroscope import Txt2VidZeroscopePlugin
from utils.video_utils import video_response


@webui(section="Video")
def add_interface(*args, **kwargs):
    async def func(
        video_model,
        prompt,
        negative_prompt,
        width,
        height,
        guidance_scale,
        num_frames,
        num_inference_steps,
        fps,
        seed,
        interpolateFilm,
        interpolateRife,        
        fast_interpolate,
        audio,
    ):
        if video_model == "Zeroscope":
            plugin_type = Txt2VidZeroscopePlugin
        elif video_model == "AnimateLCM":
            plugin_type = Txt2VidAnimatePlugin
        elif video_model == "Zero":
            plugin_type = Txt2VidZeroPlugin
        else:
            raise ValueError(f"Unknown video model: {video_model}")

        plugin = await use_plugin(plugin_type)
        frames = await plugin.generate(
            Txt2VidRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                fps=fps,
                seed=seed,
                interpolateFilm=interpolateFilm,
                interpolateRife=interpolateRife,
                fast_interpolate=fast_interpolate,
                audio=audio,
            )
        )
        file_path = video_response(
            None, frames, fps, interpolateFilm, interpolateRife, fast_interpolate, audio, True
        )
        release_plugin(plugin_type)
        return gr.Video(file_path, width=width, height=height, label="Video Output")
        # file_path = random_filename("mp4")
        # save_video_from_frames(frames, file_path, fps)
        # return file_path

    tab = gr.Tab(
        label="Text-to-Video",
    )

    with tab:
        with gr.Row():
            with gr.Column():
                grModel = gr.Radio(
                    label="Model",
                    choices=["AnimateLCM", "Zeroscope", "Zero"],
                    value="AnimateLCM",
                )
                grPrompt = gr.TextArea(label="Prompt", lines=3, value="a beautiful forest scene")
                grNegativePrompt = gr.TextArea(label="Negative Prompt", lines=3, value="low quality")

                with gr.Accordion(label="Settings"):
                    grWidth = gr.Slider(
                        label="Width", minimum=256, maximum=1024, value=512, step=64
                    )
                    grHeight = gr.Slider(
                        label="Height", minimum=256, maximum=1024, value=384, step=64
                    )
                    grGuidanceScale = gr.Slider(
                        label="Guidance Scale", minimum=1.0, maximum=10.0, value=2.0
                    )
                    grNumFrames = gr.Slider(
                        label="Number of Frames",
                        minimum=1,
                        maximum=100,
                        value=17,
                        precision=0,
                    )
                    grNumInferenceSteps = gr.Slider(
                        label="Number of Inference Steps",
                        minimum=1,
                        maximum=100,
                        value=6,
                        precision=0,
                    )
                    grFPS = gr.Slider(
                        label="Frames per Second", minimum=1, maximum=60, value=12
                    )
                    grSeed = gr.Slider(
                        label="Seed", minimum=-1, maximum=100, value=-1, precision=0,
                    )
                    grAudio = gr.Textbox(label="Audio Path or URL")
            with gr.Column():
                grVideoOutput = gr.Video(
                    label="Video Output",
                    height=512,
                )
                with gr.Row():
                    grInterpolateFilm = gr.Number(
                        label="Interpolate (FiLM)",
                        minimum=0,
                        maximum=3,
                        value=1,
                        precision=0,
                    )
                    grInterpolateRife = gr.Number(
                        label="Interpolate (RIFE)",
                        minimum=0,
                        maximum=3,
                        value=1,                        
                        precision=0,
                    )
                    grFastInterpolate = gr.Checkbox(
                        label="Fast Interpolate", value=True
                    )
                grButton = gr.Button("Generate Video")

        grButton.click(
            func,
            inputs=[
                grModel,
                grPrompt,
                grNegativePrompt,
                grWidth,
                grHeight,
                grGuidanceScale,
                grNumFrames,
                grNumInferenceSteps,
                grFPS,
                grSeed,
                grInterpolateFilm,
                grInterpolateRife,
                grFastInterpolate,
                grAudio,
            ],
            outputs=grVideoOutput,
        )

    return tab


add_interface()
