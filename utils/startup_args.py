import argparse
import sys
from settings import HOST, LLM_MODEL, PORT, SD_MODEL, TTS_MODEL


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

    startup_args = parser.parse_args()


def print_help():
    return parser.print_help()
