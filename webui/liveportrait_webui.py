import logging
import gradio as gr
from modules.webui import webui
from modules.plugins import use_plugin_unsafe
from plugins.img2vid_liveportrait import (
    Img2VidLivePortraitPlugin,
    Img2VidLivePortraitRequest,
)
from utils.file_utils import random_filename
from utils.video_utils import remove_audio, replace_audio

video_path = None
audio_path = None


@webui()
def add_interface(*args, **kwargs):

    async def remux():
        new_file = random_filename("mp4")        
        return replace_audio(video_path, audio_path, new_file)

    async def func(
        image: str,
        video: str,
        relative_motion: bool,
        do_crop: bool,
        paste_back: bool,
        include_audio: bool,
        separate_audio: bool,
    ):
        global video_path
        global audio_path

        plugin: Img2VidLivePortraitPlugin = use_plugin_unsafe(Img2VidLivePortraitPlugin)

        req = Img2VidLivePortraitRequest(
            image=image,
            video=video,
            relative_motion=relative_motion,
            do_crop=do_crop,
            paste_back=paste_back,
        )

        video_path, filename = await plugin.generate(req)
        logging.info(f"Generated video: {video_path}")

        import moviepy.editor as mp

        video_clip = mp.VideoFileClip(video_path)
        audio = video_clip.audio

        audio_path = None

        if audio and separate_audio:
            audio_path = random_filename("mp3")
            audio.write_audiofile(audio_path)
            audio.close()
            audio = None

        video_clip.close()

        if not include_audio:
            video_path = remove_audio(video_path, delete_old_file=True)

        yield gr.Video(video_path, label="output", sources=None), gr.Audio(
            audio_path, label="Audio Output", interactive=True, visible=separate_audio
        )

        # delete_file(video_path)
        # if audio_path:
        #     delete_file(audio_path)

    tab = gr.Tab(
        label="LivePortrait",
    )

    with tab:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image = gr.Image(width=256, type="filepath")
                    video = gr.Video(width=256, label="Reference Video")
                with gr.Accordion(label="Audio Settings"):
                    include_audio = gr.Checkbox(True, label="Merge Original Audio")
                    separate_audio = gr.Checkbox(False, label="Output Separate Audio")
                with gr.Accordion(label="Advanced Options"):
                    relative_motion = gr.Checkbox(True, label="Relative Motion")
                    do_crop = gr.Checkbox(True, label="Crop Input")
                    paste_back = gr.Checkbox(True, label="Paste Back")
            with gr.Column():
                output = gr.Video(label="Output", sources=None)
                audio_output = gr.Audio(
                    label="Audio Output", interactive=True, visible=False
                )
                button = gr.Button("Generate Video")
                replace_audio_button = gr.Button("Replace Audio")
                replace_audio_button.click(remux)

        button.click(
            func,
            inputs=[
                image,
                video,
                relative_motion,
                do_crop,
                paste_back,
                include_audio,
                separate_audio,
            ],
            outputs=[output, audio_output],
        )


add_interface()
