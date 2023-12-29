import time
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
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from utils.console_logging import init_logging
from utils.file_utils import ensure_folder_exists
from utils.gpu_utils import load_gpu_task, set_idle_offload_time
from utils.misc_utils import print_completion_time, sys_info
from webui import launch_webui

init_logging()

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
    global start_time
    global end_time
    if end_time is None:
        end_time = print_completion_time(start_time, "Startup")


def warmup(args):
    global start_time
    logging.info("Warming up...")
    start_time = time.time()
    if args is None or args.sd:
        from clients import SDClient

        load_gpu_task("stable diffusion", SDClient, False)
        SDClient.txt2img  # still needs a load_model function
        logging.info(f"[--warmup] {SDClient.friendly_name} ready.")
    if args is None or args.tts:
        from clients import TTSClient

        load_gpu_task("tts", TTSClient, False)
        TTSClient.generate_speech("Initializing speech.")
        logging.info(f"[--warmup] {TTSClient.friendly_name} ready.")
    if args is None or args.llm:
        from clients import Exllama2Client

        load_gpu_task("exllamav2", Exllama2Client, False)
        Exllama2Client.load_model()
        logging.info(f"[--warmup] {Exllama2Client.friendly_name} ready.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_completion_time(start_time, "Warmup")


def print_urls():
    print()
    print(f"AI Assistant: http://{HOST}:{PORT}")
    print(f"Docs URL: http://{HOST}:{PORT}/api/docs")
    print(f"Swagger URL: http://{HOST}:{PORT}/api/docs/swagger")
    print()


if __name__ == "__main__":
    start_time = time.time()

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
    start_time = time.time()

    from apis.llm import llm_api
    from apis.tts import tts_api
    from apis.diffusers import diffusers_api

    app = start_fastapi()
    web_ui = launch_webui(None, prevent_thread_lock=True)
    
    tts_api(app)
    llm_api(app)
    diffusers_api(app)
    app.mount("/", StaticFiles(directory="public_html", html=True), name="static")

    #warmup(None)

    print_startup_time()

    print_urls()
