import argparse
import sys
from settings import HOST, LLM_MODEL, PORT, TTS_MODEL
from huggingface_hub import login

startup_args = None


class DefaultArgs:
    pass


if any("run:app" in arg for arg in sys.argv):
    startup_args = DefaultArgs()
    startup_args.all = True

else:
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
        help="Include diffusers Stable Diffusion support",
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
    parser.add_argument(
        "--login",
        action="store_true",
        default=False,
        help="Login to hugging face hub (required for some models)",
    )

    startup_args = parser.parse_args()

    if startup_args.get("login"):
        login()


def print_help():
    return parser.print_help()
