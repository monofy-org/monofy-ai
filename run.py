from datetime import datetime
import torch
from clients.diffusers.SDClient import SDClient
from clients.tts.TTSClient import TTSClient
from clients.llm.Exllama2Client import Exllama2Client
from settings import HOST, MEDIA_CACHE_DIR, PORT, LLM_MODEL, TTS_MODEL, SD_MODEL
import argparse
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from utils.file_utils import ensure_folder_exists
from utils.misc_utils import sys_info
from webui import launch_webui

start_time = None
end_time = None

sys_info()

ensure_folder_exists(MEDIA_CACHE_DIR)


def start_fastapi():
    return FastAPI(
        title="monofy-ai",
        description="Simple and multifaceted API for AI",
        version="0.0.1",
        redoc_url="/api/docs",
        docs_url="/api/docs/swagger",
    )


def print_startup_time():
    global end_time
    if end_time is None:
        end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    logging.info(f"Started in {round(elapsed_time,2)} seconds")


def warmup(args):
    print("Warming up...")
    if args is None or args.sd:
        SDClient.instance.txt2img  # still needs a load_model function
        print("Stable Diffusion ready.")
    if args is None or args.tts:
        TTSClient.instance.load_model()
        TTSClient.instance.generate_speech("Initializing speech.")
        print("TTS ready.")
    if args is None or args.llm:
        Exllama2Client.instance.load_model()
        print("LLM ready.")


def print_urls():
    print()
    print(f"AI Assistant: http://{HOST}:{PORT}")
    print(f"Docs URL: http://{HOST}:{PORT}/api/docs")
    print(f"Swagger URL: http://{HOST}:{PORT}/api/docs/swagger")
    print()


if __name__ == "__main__":
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="monofy-ai")

    parser.add_argument(
        "--all", action="store_true", help="Enable all features (no other flags needed)"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="FastAPI interface, supports --llm and/or --tts",
    )
    parser.add_argument(
        "--webui",
        action="store_true",
        help="Gradio interface, supports --llm and/or --tts",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        default=False,
        help=f"Include LLM support [{LLM_MODEL}]",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        default=False,
        help=f"Include TTS support [{TTS_MODEL}]",
    )
    parser.add_argument(
        "--sd",
        action="store_true",
        default=False,
        help=f"Include diffusers Stable Diffusion support [{SD_MODEL}]",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=HOST,
        help=f"The host for the FastAPI application (default: {HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"The port for the FastAPI application (default: {PORT})",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="Preload LLM, TTS, and Stable Diffusion",
    )
    args = parser.parse_args()

    if args.all:
        args.llm = True
        args.tts = True
        args.api = True
        args.webui = True
        args.sd = True
        args.warmup = True

    if not args.tts and not args.llm and not args.sd:
        parser.print_help()

    else:
        if args.all or args.warmup:
            warmup(args)

        if args.webui:
            logging.info("Launching Gradio...")
            launch_webui(
                args,
                prevent_thread_lock=args.api,
            )

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

            uvicorn.run(app, host=args.host, port=args.port)
else:
    start_time = datetime.now()

    from apis.llm import llm_api
    from apis.tts import tts_api
    from apis.diffusers import diffusers_api

    launch_webui(
        None,
        prevent_thread_lock=True,
    )
    app = start_fastapi()
    tts_api(app)
    llm_api(app)
    diffusers_api(app)
    app.mount("/", StaticFiles(directory="public_html", html=True), name="static")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    warmup(None)

    print_startup_time()

    print_urls()
