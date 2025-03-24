import io
import gradio as gr
import numpy as np
from modules.plugins import use_plugin_unsafe
from modules.webui import webui
from plugins.txt2wav_musicgen import Txt2WavMusicGenPlugin, MusicGenRequest
from utils.gpu_utils import random_seed_number


empty_ndarray = np.array([], dtype=np.int16)


@webui()
def add_interface(*args, **kwargs):

    def func(
        prompt,
        duration,
        guidance_scale,
        temperature,
        top_p,
        streaming,
        prebuffer_chunks,
        seed_mode,
        seed_number,
    ):

        seed = seed_number if seed_mode == "Fixed" else random_seed_number()

        req = MusicGenRequest(
            prompt=prompt,
            duration=duration,
            streaming=streaming,
            guidance_scale=guidance_scale,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        plugin = None

        plugin: Txt2WavMusicGenPlugin = use_plugin_unsafe(Txt2WavMusicGenPlugin)

        buffer = []
        full_buffers = []
        rebuffer_chunks = 3

        for chunk in plugin.generate(req):
            if len(buffer) < prebuffer_chunks:
                full_buffers.append(chunk)
                buffer.append(chunk)
                continue
            if len(buffer) == prebuffer_chunks:
                prebuffer_chunks = rebuffer_chunks
                for b in buffer:
                    yield seed, b, None
                buffer = []
            yield seed, chunk, None

        if len(buffer) > 0:
            for b in buffer:
                yield seed, b, None

        wav_ndarray = np.concatenate(full_buffers)
        wav_ndarray = wav_ndarray.astype(np.int16)

        yield seed, None, (32000, wav_ndarray)

    tab = gr.Tab(label="Musicgen")

    with tab:
        prompt = gr.Textbox(
            label="Prompt",
            value="90s hip-hop, smooth bass, clean drums, downpitched samples",
        )
        duration = gr.Slider(
            label="Duration",
            value=10,
            minimum=1,
            maximum=30,
            step=0.1,
        )

        guidance_scale = gr.Slider(
            label="Guidance Scale",
            value=6.5,
            minimum=0.1,
            maximum=20.0,
            step=0.1,
        )

        temperature = gr.Slider(
            label="Temperature",
            value=1,
            minimum=0.1,
            maximum=2.0,
            step=0.01,
        )

        top_p = gr.Slider(
            label="Top P",
            value=0.6,
            minimum=0.1,
            maximum=1.0,
            step=0.01,
        )

        with gr.Row():
            seed_mode = gr.Radio(["Random", "Fixed"], value="Random", label="Seed")
            seed_number = gr.Number(
                -1,
                maximum=2**64 - 1,
                minimum=-1,
                precision=0,
                label="Seed Number",
                elem_id="seed_number",
            )

        with gr.Row():
            streaming = gr.Checkbox(
                label="Streaming (experimental)",
                value=False,
            )
            prebuffer_chunks = gr.Number(
                label="Prebuffer chunks",
                minimum=1,
                maximum=10,
                value=6,
                step=1,
                precision=0,
            )

        button: gr.Button = gr.Button("Generate Audio")

        audio = gr.Audio(
            label="Generated Audio",
            format="wav",
            streaming=True,
            interactive=False,
            autoplay=True,
        )

        audio_output = gr.Audio()

        button.click(
            func,
            [
                prompt,
                duration,
                guidance_scale,
                temperature,
                top_p,
                streaming,
                prebuffer_chunks,
                seed_mode,
                seed_number,
            ],
            outputs=[seed_number, audio, audio_output],
        )


add_interface()
