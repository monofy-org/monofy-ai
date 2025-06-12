from io import BytesIO
import json
import logging
import os
from typing import Literal, Optional

from fastapi import File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pedalboard import ExternalPlugin, Pedalboard, load_plugin
from pedalboard.io import AudioFile
from pydantic import BaseModel

from modules.plugins import router
from utils.audio_utils import get_audio_from_request, wav_to_mp3
from utils.file_utils import random_filename


class ProcessAudioRequest(BaseModel):
    plugins: list
    audio: str
    format: Optional[Literal["wav", "mp3"]] = "wav"


class ProcessAudioParamsRequest(BaseModel):
    plugin: str
    show_details: Optional[bool] = False
    show_ranges: Optional[bool] = False
    configure: Optional[bool] = False    


def load_preset(plugin, preset_path):
    logging.info("Using preset: " + preset_path)

    with open(preset_path, "r") as f:
        preset: dict = json.load(f)

    for parameter_name, serialized_value in preset.items():
        setattr(plugin, parameter_name, serialized_value)


def get_plugin(plugin_name: str):
    plugin_path: str

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

    preset_path = "vstplugins/presets/" + os.path.basename(plugin_path) + ".json"
    if os.path.exists(preset_path):
        load_preset(plugin, preset_path)

    return plugin, plugin_path


def get_plugin_name(plugin: ExternalPlugin, plugin_path: str):
    names = plugin.get_plugin_names_for_file(plugin_path)
    print(", ".join(names))


@router.post("/process-audio/params")
async def process_audio_params(req: ProcessAudioParamsRequest):
    plugin: ExternalPlugin
    plugin, path = get_plugin(req.plugin)

    if req.configure:
        plugin.show_editor()

        param_value_dict = {
            parameter_name: getattr(plugin, parameter_name)
            for parameter_name in plugin.parameters.keys()
        }
        from pedalboard._pedalboard import WrappedBool

        param_value_dict = {
            k: (bool(v) if isinstance(v, WrappedBool) else v)
            for k, v in param_value_dict.items()
        }

        # remove all entries with a value of "--"
        param_value_dict = {
            k: v for k, v in param_value_dict.items() if k != "program" and v != "--"
        }

        os.makedirs("vstplugins/presets", exist_ok=True)
        filename = os.path.join(
            "vstplugins", "presets", os.path.basename(path) + ".json"
        )

        with open(filename, "w") as f:
            json.dump(param_value_dict, f)
        return FileResponse(filename, media_type="application/json")

    if not req.show_details and not req.show_ranges:
        return [param for param in plugin.parameters.keys()]

    params = []
    for key in plugin.parameters.keys():
        param = plugin.parameters[key]

        value = (
            param.raw_value
            if param.name != "Program"
            else str(param).split('value="')[1].split('"')[0]
        )

        result: dict = {
            "name": key,
            "value": float(0) if value in [float("-inf"), float("inf")] else value,
        }

        if req.show_ranges:
            # Handle infinity and NaN values
            result["min"] = (
                float("-1e308")
                if param.range[0] in [float("-inf"), float("inf")]
                else param.range[0]
            )
            result["max"] = (
                float("1e308")
                if param.range[1] in [float("-inf"), float("inf")]
                else param.range[1]
            )
            result["step"] = (
                float(0)
                if param.range[2] in [float("-inf"), float("inf")]
                else param.range[2]
            )

        params.append(result)

    return JSONResponse(params)


@router.post("/process-audio")
async def process_audio(req: ProcessAudioRequest):    
    output_path = await process(req)
    return FileResponse(
        output_path,
        media_type="audio/mpeg" if req.format == "mp3" else "audio/wav",
        filename=os.path.basename(output_path),
    )

async def process(req: ProcessAudioRequest):
    input_path = get_audio_from_request(req.audio)
    output_path = random_filename(req.format)

    plugins: list[ExternalPlugin] = []
    for plugin_name in req.plugins:
        plugin, _ = get_plugin(plugin_name)
        plugins.append(plugin)

    # params = plugin.parameters.keys()

    # for param in params:
    #     print(param, plugin.parameters[param])

    board = Pedalboard(plugins)

    # Load audio
    with AudioFile(input_path, "r") as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate
    # Process audio
    effected = board(audio, samplerate)

    # Save output
    with AudioFile(output_path, "w", samplerate, effected.shape[0]) as f:
        f.write(effected)

    if req.format == "mp3":
        bytesio = BytesIO()
        with open(output_path, "rb") as f:
            bytesio.write(f.read())
        bytesio.seek(0)
        effected = wav_to_mp3(bytesio, samplerate)

    return output_path


@router.post("/process-audio-form")
async def process_audio_form(plugin_name: str, file: UploadFile = File(...)):
    plugin, _ = get_plugin(plugin_name)

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
