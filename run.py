from datetime import datetime
import torch
from utils.startup_args import print_help, startup_args as args
from settings import (
    HOST,
    IDLE_OFFLOAD_TIME,
    MEDIA_CACHE_DIR,
    PORT,
)

import logging
import uvicorn
import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from utils.file_utils import ensure_folder_exists
from utils.gpu_utils import free_vram, set_idle_offload_time
from utils.misc_utils import sys_info
from webui import launch_webui


logging.basicConfig(level=logging.INFO)

start_time = None
end_time = None

sys_info()

ensure_folder_exists(MEDIA_CACHE_DIR)


def start_fastapi():
    app = FastAPI(
        title="monofy-ai",
        description="Simple and multifaceted API for AI",
        version="0.0.1",
        redoc_url="/api/docs",
        docs_url="/api/docs/swagger",
    )

    set_idle_offload_time(IDLE_OFFLOAD_TIME)

    return app


def print_startup_time():
    global end_time
    if end_time is None:
        end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    print()
    logging.info(f"Startup completed in {round(elapsed_time,2)} seconds")


def warmup(args):
    print("Warming up...")
    if args is None or args.sd:
        from clients import SDClient
        free_vram("stable diffusion", SDClient)
        SDClient.txt2img  # still needs a load_model function
        logging.info("[--warmup] Stable Diffusion ready.")
    if args is None or args.tts:
        from clients import TTSClient
        free_vram("tts", TTSClient)
        TTSClient.load_model()
        TTSClient.generate_speech("Initializing speech.")
        logging.info("[--warmup] TTS ready.")
    if args is None or args.llm:
        from clients import Exllama2Client
        free_vram("exllamav2", Exllama2Client)
        Exllama2Client.load_model()
        logging.info("[--warmup] LLM ready.")
    if torch.cuda.is_available:
        torch.cuda.empty_cache()


def print_urls():
    print()
    print(f"AI Assistant: http://{HOST}:{PORT}")
    print(f"Docs URL: http://{HOST}:{PORT}/api/docs")
    print(f"Swagger URL: http://{HOST}:{PORT}/api/docs/swagger")
    print()


if __name__ == "__main__":
    
    start_time = datetime.now()

    if args.all:        
        args.llm = True
        args.tts = True
        args.api = True
        args.webui = True
        args.sd = True
        args.warmup = True        

    if not args.tts and not args.llm and not args.sd:
        print_help()

    else:
        if args.all or args.warmup:
            warmup(args)

        if args.api:
            app = start_fastapi()

        if args.webui:
            logging.info("Launching Gradio...")
            web_ui = launch_webui(args, prevent_thread_lock=args.api)

        if args.api:
            logging.info("Launching FastAPI...")

            app = start_fastapi()
            if args.webui:
                app = gr.mount_gradio_app(app, web_ui, path="/gradio")

            if args.sd:
                from apis.diffusers import diffusers_api

                diffusers_api(app)

            if args.llm:
                from apis.llm import llm_api

                llm_api(app)

            if args.tts:
                from apis.tts import tts_api

                tts_api(app)

            app.mount(
                "/", StaticFiles(directory="public_html", html=True), name="static"
            )

            print_startup_time()

            print_urls()

            uvicorn.run(
                app,
                host=args.host or HOST,
                port=args.port or PORT,
            )
else:
    start_time = datetime.now()

    from apis.llm import llm_api
    from apis.tts import tts_api
    from apis.diffusers import diffusers_api

    app = start_fastapi()
    web_ui = launch_webui(None, prevent_thread_lock=True)
    app = gr.mount_gradio_app(app, web_ui, path="/gradio")
    tts_api(app)
    llm_api(app)
    diffusers_api(app)
    app.mount("/", StaticFiles(directory="public_html", html=True), name="static")

    warmup(None)

    print_startup_time()

    print_urls()
