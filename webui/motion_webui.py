import gradio as gr
from modules.webui import webui
from modules.plugins import use_plugin_unsafe
from plugins.vid2densepose import Vid2DensePosePlugin
from plugins.experimental.vid2vid_magicanimate import Vid2VidMagicAnimatePlugin


@webui()
def add_interface(*args, **kwargs):

    async def extract_motion(reference_video: str):
        if not reference_video:
            raise gr.Error("Please upload a reference video")
        plugin: Vid2DensePosePlugin = use_plugin_unsafe(Vid2DensePosePlugin)
        return plugin.generate(reference_video)

    async def func(image, video, motion_video, additional_audio):
        plugin: Vid2VidMagicAnimatePlugin = use_plugin_unsafe(Vid2VidMagicAnimatePlugin)
        video_path = plugin.generate(image, motion_video)
        yield gr.Video(video_path, label="output", sources=None)

    tab = gr.Tab(
        label="Motion Transfer",
    )

    with tab:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        original_video = gr.Video(width=256, label="Original Video")
                        extract_button = gr.Button("Extract Motion")
                    motion_video = gr.Video(
                        width=256, label="Motion Data", sources="upload"
                    )

                with gr.Row():
                    image = gr.Image(type="filepath")
                with gr.Accordion(label="Audio Settings"):
                    include_audio = gr.Checkbox(True, label="Merge Original Audio")
                    additional_audio = gr.Audio(
                        label="Additional Audio", interactive=True
                    )

            extract_button.click(
                extract_motion, inputs=[original_video], outputs=[motion_video]
            )

            with gr.Column():
                output = gr.Video(label="Output", sources=None)
                button = gr.Button("Generate Video")

        button.click(
            func,
            inputs=[
                image,
                original_video,
                motion_video,
                additional_audio,
            ],
            outputs=[output],
        )


add_interface()
