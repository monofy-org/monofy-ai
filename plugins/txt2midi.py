import logging
import os
from typing import Optional

import torch
from fastapi import HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from transformers import pipeline

from modules.plugins import PluginBase, release_plugin, use_plugin
from utils.file_utils import random_filename


class Txt2MidiRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 1000
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0


class Txt2MidiPlugin(PluginBase):
    name = "Txt2Midi (llama-midi)"
    description = "Text-to-MIDI using llama-midi"

    def load_model(self):
        pipe = self.resources.get("pipeline")
        if pipe:
            return pipe

        pipe = pipeline(
            "text-generation",
            model="dx2102/llama-midi",
            torch_dtype=self.dtype,
            device=self.device,  # cuda/mps/cpu
        )

        self.resources["pipe"] = pipe

        return pipe

    def generate(self, req: Txt2MidiRequest):
        pipe = self.load_model()
        result = pipe(
            req.prompt,
            max_length=req.max_length,
            temperature=req.temperature,
            top_p=req.top_p,
        )[0]["generated_text"]        
        output_path = random_filename("mid")
        self.postprocess(result, output_path)

        return output_path

    def postprocess(self, txt: str, path: str):
        # assert txt.startswith(prompt)
        txt = txt.split("\n\n")[-1]

        tracks = {}

        now = 0
        # we need to ignore the invalid output by the model
        try:
            import symusic

            for line in txt.split("\n"):
                pitch, duration, wait, velocity, instrument = line.split()
                pitch, duration, wait, velocity = [
                    int(x) for x in [pitch, duration, wait, velocity]
                ]
                if instrument not in tracks:
                    tracks[instrument] = symusic.core.TrackSecond()
                    if instrument != "drum":
                        tracks[instrument].program = int(instrument)
                    else:
                        tracks[instrument].is_drum = True
                # Eg. Note(time=7.47, duration=5.25, pitch=43, velocity=64, ttype='Second')
                tracks[instrument].notes.append(
                    symusic.core.NoteSecond(
                        time=now / 1000,
                        duration=duration / 1000,
                        pitch=int(pitch),
                        velocity=int(velocity * 4),
                    )
                )
                now += wait
        except Exception as e:
            print("Postprocess: Ignored error:", e)

        print(
            f"Postprocess: Got {sum(len(track.notes) for track in tracks.values())} notes"
        )

        try:
            import symusic

            score = symusic.Score(ttype="Second")
            score.tracks.extend(tracks.values())
            score.dump_midi(path)
        except Exception as e:
            print("Postprocess: Ignored postprocessing error:", e)


@PluginBase.router.post("/txt2midi")
async def txt2midi(req: Txt2MidiRequest):
    plugin: Txt2MidiPlugin = None
    try:
        plugin = await use_plugin(Txt2MidiPlugin)
        midi_path = plugin.generate(req)
        return FileResponse(
            midi_path, filename=os.path.basename(midi_path), media_type="audio/midi"
        )
    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(500, "Generation failed")
    finally:
        if plugin:
            release_plugin(Txt2MidiPlugin)
