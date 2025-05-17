import json
import logging
import os
from typing import Literal, Optional

from fastapi import BackgroundTasks, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel

from modules.plugins import PluginBase, release_plugin, use_plugin
from plugins.extras.lyrics import generate_lyrics
from utils.console_logging import log_loading
from utils.file_utils import random_filename
from utils.gpu_utils import autodetect_dtype, clear_gpu_cache, random_seed_number


class Txt2WavACEStepRequest(BaseModel):
    prompt: str
    format: Literal["wav", "mp3"] = "mp3"
    negative_prompt: Optional[str] = None
    lyrics: Optional[str] = None
    lyrics_prompt: Optional[str] = None
    audio_duration: Optional[float] = 120
    seed: Optional[int] = -1
    smart: Optional[bool] = True
    guidance_scale: Optional[float] = 15.0
    guidance_scale_text: Optional[float] = 0.0
    guidance_scale_lyric: Optional[float] = 0.0
    guidance_interval: Optional[float] = 0.5
    guidance_interval_decay: Optional[float] = 0.0
    min_guidance_scale: Optional[float] = 3.0
    num_inference_steps: Optional[int] = 60
    scheduler_type: Optional[str] = "euler"
    cfg_type: Optional[str] = "apg"
    omega_scale: Optional[float] = 10.0
    # actual_seeds: Optional[list[int]] = None
    oss_steps: Optional[list[int]] = None
    use_erg_tag: Optional[bool] = True
    use_erg_lyric: Optional[bool] = True
    use_erg_diffusion: Optional[bool] = True


def _smart_prompt(req: Txt2WavACEStepRequest) -> str:
    file = "./user-settings/ace-step.json"
    if not os.path.exists(file):
        logging.warning(f"{file} not found. Using default prompt.")
        return
    
    try:
        data = open(file, "r").read()
        genre_prompts: dict = json.loads(data)
    except Exception as e:
        logging.error(e)
        return

    genre: str
    added_prompts = []
    added_negatives = []

    # Check if any genre keywords are in the prompt
    for genre, prompts in genre_prompts.items():
        if genre.lower() in req.prompt.lower():
            # Append positive prompts
            added_prompts.extend(prompts["prompt"])
            req.prompt = req.prompt + ", " + ", ".join(prompts["prompt"])

            # Initialize negative prompt if None
            if req.negative_prompt is None:
                req.negative_prompt = ""

            # Append negative prompts
            if req.negative_prompt:
                req.negative_prompt += ", "
            added_negatives.extend(prompts["negative_prompt"])
            req.negative_prompt += ", ".join(prompts["negative_prompt"])
            continue

    logging.info(f"Smart prompt added positive terms: {', '.join(added_prompts)}")
    logging.info(f"Smart prompt added negative terms: {', '.join(added_negatives)}")

    return req.prompt


class Txt2WavACEStepPlugin(PluginBase):
    name = "Txt2Wav (ACE-Step)"
    description = "Text-to-wav using ACE-Step"
    instance = None

    def __init__(self):
        super().__init__()

        self.dtype = autodetect_dtype(True)

    def load_model(self):
        log_loading("ACE-Step", "hf cache")

        if self.resources.get("pipeline"):
            return self.resources["pipeline"]

        from submodules.ACE_Step.acestep.pipeline_ace_step import ACEStepPipeline
        pipeline: ACEStepPipeline = ACEStepPipeline(
            checkpoint_dir="",
            dtype="bfloat16",
            torch_compile=False,
        )

        self.resources["pipeline"] = pipeline

        return pipeline

    async def generate(self, req: Txt2WavACEStepRequest):
        if req.smart:
            _smart_prompt(req)

        if req.lyrics_prompt:
            req.lyrics = generate_lyrics(
                req.prompt, req.lyrics_prompt, unload_after=True
            )

        pipe = self.load_model()

        if req.seed == -1:
            req.seed = random_seed_number()

        seed = req.seed if req.seed > -1 else random_seed_number()

        output_path = random_filename(req.format)

        pipe(
            req.audio_duration,
            req.prompt,
            req.lyrics,
            req.num_inference_steps,
            req.guidance_scale,
            req.scheduler_type,
            req.cfg_type,
            req.omega_scale,
            [seed],
            req.guidance_interval,
            req.guidance_interval_decay,
            req.min_guidance_scale,
            req.use_erg_tag,
            req.use_erg_lyric,
            req.use_erg_diffusion,
            req.oss_steps,
            req.guidance_scale_text,
            req.guidance_scale_lyric,
            save_path=output_path,
            format=req.format,
        )

        return output_path


@PluginBase.router.post("/txt2wav/ace-step", tags=["Audio and Music"])
async def txt2wav_ace_step(
    background_tasks: BackgroundTasks, req: Txt2WavACEStepRequest
):
    plugin: Txt2WavACEStepPlugin = None
    try:
        plugin = await use_plugin(Txt2WavACEStepPlugin)
        output_path = await plugin.generate(req)
        return FileResponse(output_path, media_type="audio/wav")
    except Exception as e:
        logging.exception(e)
        raise e
    finally:
        if plugin is not None:
            release_plugin(plugin)
            clear_gpu_cache()


@PluginBase.router.get("/txt2wav/ace-step", tags=["Audio and Music"])
async def txt2wav_ace_step_get(req: Txt2WavACEStepRequest = Depends()):
    return await txt2wav_ace_step(req)
