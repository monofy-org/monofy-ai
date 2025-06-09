import json
import logging
import os
from typing import Optional

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pedalboard import ExternalPlugin, Pedalboard, load_plugin
from pedalboard.io import AudioFile
from pydantic import BaseModel

from modules.plugins import router
from utils.audio_utils import get_audio_from_request
from utils.file_utils import random_filename


class ProcessAudioRequest(BaseModel):
    plugin_name: str
    audio: str


class ProcessAudioParamsRequest(BaseModel):
    plugin_name: str
    configure: Optional[bool] = False

def get_plugin(plugin_name: str):
    with open("user-settings/vst3au.json", "r") as f:
        plugins: dict = json.load(f)
        if not plugins:
            raise HTTPException(500, "No plugins configured")

        plugin_details: dict = plugins.get(plugin_name)
        if not plugin_details:
            raise HTTPException(400, "Plugin not found: " + plugin_name)

        plugin_path = plugin_details.get("path")
        if not plugin_path:
            raise HTTPException(400, "Plugin path not defined")

        if not os.path.exists(plugin_path):
            raise HTTPException(500, "Plugin not installed: " + plugin_path)

    logging.info("Using audio plugin: " + plugin_path)

    plugin = load_plugin(plugin_path)

    print(", ".join(plugin.get_plugin_names_for_file(plugin_path)))

    return plugin


@router.post("/process-audio/params")
async def process_audio_params(req: ProcessAudioParamsRequest):
    plugin: ExternalPlugin = get_plugin(req.plugin_name)

    if req.configure:
        plugin.show_editor()
    
    params = []
    for key in plugin.parameters.keys():        
        param = plugin.parameters[key]
        print(param)
        value = (
            param.raw_value
            if param.name != "Program"
            else str(param).split('value="')[1].split('"')[0]
        )
        params.append(
            {
                "name": param.name,
                "value": value,
                "min": param.range[0],
                "max": param.range[1],
                "step": param.range[2],
            }
        )
        print(params)
    return JSONResponse(params)


@router.post("/process-audio")
async def process_audio(req: ProcessAudioRequest):
    input_path = get_audio_from_request(req.audio)
    output_path = random_filename("wav")
    plugin: ExternalPlugin = get_plugin(req.plugin_name)

    params = plugin.parameters.keys()

    for param in params:
        print(param, plugin.parameters[param])

    board = Pedalboard([plugin])

    # Load audio
    with AudioFile(input_path, "r") as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate
    # Process audio
    effected = board(audio, samplerate)
    # Save output
    with AudioFile(output_path, "w", samplerate, effected.shape[0]) as f:
        f.write(effected)

    return FileResponse(
        output_path, media_type="audio/wav", filename=os.path.basename(output_path)
    )


@router.post("/process-audio-form")
async def process_audio_form(plugin_name: str, file: UploadFile = File(...)):
    plugin = get_plugin(plugin_name)

    input_path = random_filename("wav")
    output_path = random_filename("wav")

    # Save uploaded file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Load audio
    with AudioFile(input_path, "r") as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate

    board = Pedalboard([plugin])
    # Process audio
    effected = board(audio, samplerate)

    # Save output
    with AudioFile(output_path, "w", samplerate, effected.shape[0]) as f:
        f.write(effected)

    # Clean up input file
    os.remove(input_path)

    return FileResponse(
        output_path, media_type="audio/wav", filename=os.path.basename(output_path)
    )
