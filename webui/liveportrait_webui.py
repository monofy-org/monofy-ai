import io
import logging
import gradio as gr
from modules.webui import webui
from modules.plugins import use_plugin_unsafe
from plugins.img2vid_liveportrait import (
    Img2VidLivePortraitPlugin,
    Img2VidLivePortraitRequest,
)
from utils.file_utils import delete_file, random_filename


@webui()
def add_interface(*args, **kwargs):
    async def func(
        image: str,
        video: str,
        relative_motion: bool,
        do_crop: bool,
        paste_back: bool,
    ):
        plugin: Img2VidLivePortraitPlugin = use_plugin_unsafe(Img2VidLivePortraitPlugin)

        req = Img2VidLivePortraitRequest(
            image=image,
            video=video,
            relative_motion=relative_motion,
            do_crop=do_crop,
            paste_back=paste_back,
        )

        full_path, filename = await plugin.generate(req)
        logging.info(f"Generated video: {full_path}")

        import moviepy.editor as mp

        video_clip = mp.VideoFileClip(full_path)
        audio = video_clip.audio        
        # write to gradio temp file
        audio_path = random_filename("mp3")
        audio.write_audiofile(audio_path)
        audio.close()      

        yield gr.Video(full_path, label="output"), gr.Audio(
            audio_path, label="Audio Output", interactive=False
        )

        delete_file(full_path)

    tab = gr.Tab(
        label="LivePortrait",
    )

    with tab:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image = gr.Image(width=256, type="filepath")
                    video = gr.Video(width=256, label="Reference Video")
                with gr.Accordion(label="Advanced Options"):
                    relative_motion = gr.Checkbox(True, label="Relative Motion")
                    do_crop = gr.Checkbox(True, label="Crop Input")
                    paste_back = gr.Checkbox(True, label="Paste Back")
            with gr.Column():
                output = gr.Video(label="Output")
                audio_output = gr.Audio(label="Audio Output", interactive=False)
                button = gr.Button("Generate Video")

        button.click(
            func,
            inputs=[image, video, relative_motion, do_crop, paste_back],
            outputs=[output, audio_output],
        )


add_interface()
