import gradio as gr

from classes.requests import Txt2VidRequest
from modules.plugins import release_plugin, use_plugin
from modules.webui import webui
from plugins.txt2vid import Txt2VidZeroPlugin
from plugins.txt2vid_animate import Txt2VidAnimatePlugin
from plugins.txt2vid_zeroscope import Txt2VidZeroscopePlugin
from plugins.video_plugin import VideoPlugin
from settings import SD_DEFAULT_MODEL_INDEX, SD_MODELS


@webui()
def add_interface(*args, **kwargs):
    async def func(
        model_path,
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
        interpolate_film,
        interpolate_rife,
        fast_interpolate,
        audio,
    ):
        plugin_type: type[VideoPlugin] = None
        if video_model == "Zeroscope":
            plugin_type = Txt2VidZeroscopePlugin
        elif video_model == "AnimateLCM":
            plugin_type = Txt2VidAnimatePlugin
        elif video_model == "Zero":
            plugin_type = Txt2VidZeroPlugin
        else:
            raise ValueError(f"Unknown video model: {video_model}")

        model_index = SD_MODELS.index(model_path)

        plugin: VideoPlugin = None
        try:
            plugin = await use_plugin(plugin_type)
            frames = await plugin.generate(
                Txt2VidRequest(
                    model_index=model_index,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    fps=fps,
                    seed=seed,
                    interpolateFilm=interpolate_film,
                    interpolateRife=interpolate_rife,
                    fast_interpolate=fast_interpolate,
                    audio=audio,
                )
            )
            file_path = plugin.video_response(
                None,
                frames,
                Txt2VidRequest(
                    interpolate_film=interpolate_film,
                    interpolate_rife=interpolate_rife,
                    fast_interpolate=fast_interpolate,
                    fps=fps,
                    audio=audio,
                ),
                True,
            )
        except Exception as e:
            raise e
        finally:
            if plugin:
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
                grModel = gr.Dropdown(
                    SD_MODELS,
                    label="Model",
                    value=SD_MODELS[SD_DEFAULT_MODEL_INDEX],
                )
                grModule = gr.Radio(
                    label="Motion Module",
                    choices=["AnimateLCM", "Zeroscope", "Zero"],
                    value="AnimateLCM",
                )
                grPrompt = gr.TextArea(
                    label="Prompt",
                    lines=3,
                    value="sci-fi movie scene, humanoid cyborg robot walking, third-person view",
                )
                grNegativePrompt = gr.TextArea(
                    label="Negative Prompt",
                    lines=3,
                    value="nsfw, blurry, deformed, worst quality, dark shadows, bright lights, bloom",
                )

                with gr.Accordion(label="Settings"):
                    with gr.Row():
                        grWidth = gr.Slider(
                            label="Width", minimum=256, maximum=1024, value=512, step=64
                        )
                        grHeight = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64,
                        )
                    grGuidanceScale = gr.Slider(
                        label="Guidance Scale", minimum=1.0, maximum=10.0, value=1.0
                    )
                    grNumInferenceSteps = gr.Slider(
                        label="Number of Inference Steps",
                        minimum=1,
                        maximum=30,
                        value=4,
                        step=1,
                    )
                    grNumFrames = gr.Slider(
                        label="Number of Frames",
                        minimum=1,
                        maximum=100,
                        value=16,
                        step=1,
                    )
                    grSeed = gr.Number(
                        label="Seed",
                        minimum=-1,
                        maximum=2**64 - 1,
                        value=-1,
                        precision=0,
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
                    grFPS = gr.Number(
                        label="Output FPS", minimum=1, maximum=60, value=24, precision=0
                    )
                grButton = gr.Button("Generate Video")

        grButton.click(
            func,
            inputs=[
                grModel,
                grModule,
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
