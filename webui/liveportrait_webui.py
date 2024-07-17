import gradio as gr
from modules.webui import webui
from modules.plugins import use_plugin_unsafe
from plugins.img2vid_liveportrait import (
    Img2VidLivePortraitPlugin,
    Img2VidLivePortraitRequest,
)


@webui()
def add_interface(*args, **kwargs):
    async def func(
        image: str,
        video: str,
    ):
        plugin: Img2VidLivePortraitPlugin = use_plugin_unsafe(Img2VidLivePortraitPlugin)

        req = Img2VidLivePortraitRequest(
            image=image,
            video=video,
            relative_motion=True,
            do_crop=True,
            paste_back=True,
        )

        full_path, filename = await plugin.generate(req)
        return gr.Video(full_path, label="output")

    tab = gr.Tab(
        label="LivePortrait",
    )

    with tab:
        with gr.Column():
            with gr.Row():
                image = gr.Image(width=256, type="filepath")
                video = gr.Video(width=256, label="Reference Video")

            button = gr.Button("Generate Video")
        with gr.Column():
            output = gr.Video(label="Output")

        button.click(func, inputs=[image, video], outputs=[output])

add_interface()
