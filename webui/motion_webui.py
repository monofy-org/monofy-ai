import gradio as gr
from modules.webui import webui
from modules.plugins import release_plugin, use_plugin
from plugins.vid2densepose import Vid2DensePosePlugin
#from plugins.experimental.vid2vid_magicanimate import Vid2VidMagicAnimatePlugin
from utils.file_utils import random_filename


@webui()
def add_interface(*args, **kwargs):
    async def extract_motion(reference_video: str):
        if not reference_video:
            raise gr.Error("Please upload a reference video.")
        plugin: Vid2DensePosePlugin = None
        try:
            plugin = await use_plugin(Vid2DensePosePlugin)
            frames, fps = plugin.generate(reference_video)            
            import imageio_ffmpeg

            file_path = random_filename("mp4")
            writer = imageio_ffmpeg.write_frames(
                file_path, frames[0].shape[:2], fps=fps
            )
            writer.send(None)  # Initialize the generator
            for frame in frames:
                writer.send(frame)
            writer.close()
            yield file_path

        except Exception as e:
            raise gr.Error(str(e))
        finally:
            if plugin:
                release_plugin(plugin)

    async def func(image, video, motion_video, additional_audio):
        # if not image:
        #     raise gr.Error("Please upload an image.")
        # plugin: Vid2VidMagicAnimatePlugin = None
        # try:
        #     plugin = await use_plugin(Vid2VidMagicAnimatePlugin)
        #     video_path = plugin.generate(image, motion_video)            
        #     yield gr.Video(video_path, label="output", sources=None)
        # except Exception as e:
        #     raise gr.Error(str(e))
        # finally:
        #     if plugin:
        #         release_plugin(plugin)
        yield None

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
