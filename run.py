import os
import sys
import time
import torch
from utils.startup_args import print_help, startup_args as args
from settings import HOST, IDLE_OFFLOAD_TIME, MEDIA_CACHE_DIR, PORT, SD_USE_SDXL
import logging
import uvicorn
import warnings
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from utils.console_logging import init_logging
from utils.file_utils import ensure_folder_exists
from utils.gpu_utils import load_gpu_task, set_idle_offload_time
from utils.misc_utils import print_completion_time, sys_info
from webui import launch_webui
from utils.console_logging import show_banner


warnings.filterwarnings("ignore", category=FutureWarning)

API_PREFIX = "/api"

init_logging()

start_time = None
end_time = None

sys_info()

ensure_folder_exists(MEDIA_CACHE_DIR)

# Get the absolute path to the submodules directory
submodules_dir = os.path.abspath("submodules")

# Add the submodules directory to the Python path
for submodule in os.listdir(submodules_dir):
    submodule_path = os.path.join(submodules_dir, submodule)
    if os.path.isdir(submodule_path) and submodule_path not in sys.path:
        sys.path.insert(0, submodule_path)


def start_fastapi(args=None):
    global start_time
    start_time = time.time()

    app = FastAPI(
        title="monofy-ai",
        description="Simple and multifaceted API for AI",
        version="0.0.1",
        redoc_url="/api/docs",
        docs_url="/api/docs/swagger",
    )

    set_idle_offload_time(IDLE_OFFLOAD_TIME)

    if args is None or args.all or args.sd:
        from apis import (
            txt2img,
            img2img,
            ipadapter,
            depth,
            detect,
            vision,
            txt2vid,
            img2vid,
            shape,
            audiogen,
            musicgen,
        )

        app.include_router(txt2img.router, prefix=API_PREFIX)
        app.include_router(img2img.router, prefix=API_PREFIX)
        app.include_router(ipadapter.router, prefix=API_PREFIX)
        app.include_router(depth.router, prefix=API_PREFIX)
        app.include_router(detect.router, prefix=API_PREFIX)
        app.include_router(vision.router, prefix=API_PREFIX)
        app.include_router(txt2vid.router, prefix=API_PREFIX)
        app.include_router(img2vid.router, prefix=API_PREFIX)
        app.include_router(shape.router, prefix=API_PREFIX)
        app.include_router(audiogen.router, prefix=API_PREFIX)
        app.include_router(musicgen.router, prefix=API_PREFIX)

        if args is None or args.all or args.llm:
            from apis import llm

            app.include_router(llm.router)

        if args is None or args.all or args.tts:
            from apis import tts, whisper

            app.include_router(tts.router, prefix=API_PREFIX)
            app.include_router(whisper.router, prefix=API_PREFIX)

        app.mount("/", StaticFiles(directory="public_html", html=True), name="static")

    return app


def print_startup_time():
    global start_time
    global end_time
    if end_time is None:
        end_time = print_completion_time(start_time, "Startup")


def warmup(args):
    logging.info("Warming up...")
    if args is None or args.sd:
        from clients import SDClient

        load_gpu_task("sdxl" if SD_USE_SDXL else "stable diffusion", SDClient, False)
        SDClient.pipelines["txt2img"]  # just reference something so the module loads
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


def print_urls():
    print()
    show_banner()
    print()
    logging.info(f"AI Assistant: http://{HOST}:{PORT}")
    logging.info(f"Docs URL: http://{HOST}:{PORT}/api/docs")
    logging.info(f"Swagger URL: http://{HOST}:{PORT}/api/docs/swagger")
    print()


if __name__ == "__main__":
    start_time = time.time()

    if not args.all and (
        (not args.tts and not args.llm and not args.sd)
        or (not args.api and not args.webui)
    ):
        print_help()

    else:
        if args.all or args.warmup:
            warmup(args)

        if args is None or args.all or args.webui:
            logging.info("Launching Gradio...")
            web_ui = launch_webui(args, prevent_thread_lock=args.all or args.api)

        if args is None or args.all or args.api:
            logging.info("Launching FastAPI...")

            app = start_fastapi(args)

            print_urls()

            uvicorn.run(
                app,
                host=args.host or HOST,
                port=args.port or PORT,
            )
else:
    # from apis.rignet import rignet_api
    # rignet_api(app)

    app = start_fastapi()
    web_ui = launch_webui(None, prevent_thread_lock=True)
    print_startup_time()
    print_urls()
