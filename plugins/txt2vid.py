import logging
import numpy as np
from PIL import Image
from fastapi import Depends, BackgroundTasks
from classes.requests import Txt2VidRequest
from modules.plugins import PluginBase, use_plugin, release_plugin
from utils.gpu_utils import autodetect_dtype, set_seed
from utils.video_utils import video_response


class Txt2VidZeroPlugin(PluginBase):

    name = "txt2vid"
    description = "Text-to-video generation"
    instance = None

    def __init__(self):
        import torch
        from diffusers import TextToVideoZeroPipeline

        super().__init__()

        self.chunk_size = 8

        model_id = "emilianJR/epiCRealism"

        pipe = TextToVideoZeroPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device, dtype=autodetect_dtype())

        self.resources["TextToVideoZeroPipeline"] = pipe

    async def generate(self, req: Txt2VidRequest):
        result = []
        chunk_ids = np.arange(0, req.num_frames, self.chunk_size - 1)
        req.seed, generator = set_seed(req.seed, True)
        pipe = self.resources["TextToVideoZeroPipeline"]
        for i in range(len(chunk_ids)):
            print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
            ch_start = chunk_ids[i]
            ch_end = req.num_frames if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
            # Attach the first frame for Cross Frame Attention
            frame_ids = [0] + list(range(ch_start, ch_end))
            # Fix the seed for the temporal consistency
            generator.manual_seed(req.seed)
            output = pipe(
                prompt=req.prompt,
                video_length=len(frame_ids),
                generator=generator,
                frame_ids=frame_ids,
            )
            result.append(output.images[1:])

        # Concatenate chunks and save
        result = np.concatenate(result)
        # result = [(r * 255).astype("uint8") for r in result]
        # save results as PIL images
        frames = [Image.fromarray((r * 255).astype("uint8")) for r in result]

        return frames


@PluginBase.router.post("/txt2vid/zero", tags=["Video Generation (text-to-video)"])
async def txt2vid(
    background_tasks: BackgroundTasks,
    req: Txt2VidRequest,
):
    plugin = None

    try:
        plugin: Txt2VidZeroPlugin = await use_plugin(Txt2VidZeroPlugin)
        frames = await plugin.generate(req)
        return video_response(
            background_tasks,
            frames,
            req.fps,
            req.interpolate_film,
            req.interpolate_rife,
            req.fast_interpolate,
            req.audio,
        )

    except Exception as e:
        logging.error(e, exc_info=True)
        raise e

    finally:
        if plugin:
            release_plugin(Txt2VidZeroPlugin)


@PluginBase.router.get("/txt2vid/zero", tags=["Video Generation (text-to-video)"])
async def txt2vid_from_url(
    background_tasks: BackgroundTasks,
    req: Txt2VidRequest = Depends(),
):
    return await txt2vid(background_tasks, req)
