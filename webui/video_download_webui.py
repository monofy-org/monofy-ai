import gradio as gr
from modules.webui import webui
from utils.video_utils import get_video_from_request


@webui()
def add_interface(*args, **kwargs):

    def func(url):
        result = get_video_from_request(url)
        print(result)
        return result

    tab = gr.Tab(
        label="Video DL",
    )

    with tab:
        with gr.Row():
            url_input = gr.Text(label="URL")
            url_button = gr.Button("Download")        

        video_output = gr.Video(interactive=True)

        url_button.click(func, inputs=[url_input], outputs=[video_output])


add_interface()
