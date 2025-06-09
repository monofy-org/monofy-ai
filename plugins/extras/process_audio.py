import json
import os

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pedalboard import Pedalboard, Plugin, load_plugin
from pedalboard.io import AudioFile

from utils.file_utils import random_filename

from modules.plugins import router

@router.post("/process-audio")
async def process_audio(plugin_name: str, file: UploadFile = File(...)):
    # read user-configs/vst3au.json
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

    plugin = load_plugin(plugin_path)

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

    return FileResponse(output_path, media_type="audio/wav", filename="processed.wav")
