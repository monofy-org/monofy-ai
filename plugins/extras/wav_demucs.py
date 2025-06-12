import logging
import os
import uuid
import zipfile
from typing import Literal, Optional

import demucs.separate
from fastapi import Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pydub import AudioSegment

from modules.plugins import router
from plugins.extras.process_audio import ProcessAudioRequest, process
from settings import CACHE_PATH
from utils.audio_utils import get_audio_from_request
from utils.file_utils import random_filename


class WavDemucsRequest(BaseModel):
    url: str
    stem: Optional[Literal["vocals", "drums", "bass", "other", None]] = None
    model: Optional[Literal["htdemucs", "mdx", "mdx_extra"]] = "mdx_extra"
    format: Optional[Literal["mp3", "wav"]] = "mp3"
    mp3_bitrate: Optional[int] = 192
    stem_only: Optional[bool] = False
    remaster: Optional[bool] = False
    remaster_omit: Optional[list[str]] = []


async def generate(**kwargs):
    url: str = kwargs.get("url")
    stem: str = kwargs.get("stem")
    model: str = kwargs.get("model", "mdx_extra")
    format: str = kwargs.get("format", "mp3")
    mp3_bitrate: int = kwargs.get("mp3_bitrate", 192)
    stem_only: bool = kwargs.get("stem_only", False)
    remaster: bool = kwargs.get("remaster", False)
    remaster_omit: list[str] = kwargs.get("remaster_omit", [])

    try:
        audio_path = get_audio_from_request(url)
        folder_id = str(uuid.uuid4())
        output_folder = os.path.abspath(os.path.join(CACHE_PATH, folder_id))
        os.mkdir(output_folder)

        args = []
        if remaster or format == "wav":
            format = "wav"
        else:
            format = "mp3"
            args.extend(["--mp3", "--mp3-bitrate", str(mp3_bitrate)])

        if stem:
            args.extend(["--two-stems", stem])

        args.extend(["-n", model, "-o", output_folder, audio_path])

        demucs.separate.main(args)

        if remaster:
            stems: list[str] = []

            print(os.path.join(output_folder))

            input_name_noext = os.path.basename(audio_path).rsplit(".", 1)[0]
            stem_path = os.path.join(output_folder, model, input_name_noext)

            files = os.listdir(stem_path)

            print("DEBUG 1", remaster_omit)
            print("DEBUG 2", files)

            files = [
                os.path.join(stem_path, file)
                for file in files
                if file.rsplit(".wav", 1)[0] not in remaster_omit
            ]
            for file in files:
                if file.endswith(".wav"):
                    print(f"Processing {file}")
                    stems.append(
                        await process(
                            ProcessAudioRequest(audio=file, plugins=["eq", "tube"])
                        )
                    )
                else:
                    logging.info(f"Skipping {file}")

            # recombine stems into one audio file
            output_path = random_filename(format)
            # Mix stems using pydub

            mixed: AudioSegment = AudioSegment.from_file(stems[0])

            for stem in stems[1:]:
                segment = AudioSegment.from_file(stem)
                mixed = mixed.overlay(segment)

            mixed.export(output_path, format=format)

            return FileResponse(
                output_path,
                media_type="audio/mpeg" if format == "mp3" else "audio/wav",
            )

        elif stem_only:
            stem_parent = stem_path = os.path.join(output_folder, model)
            stem_parent = os.path.join(stem_parent, os.listdir(stem_parent)[0])
            stem_path = os.path.join(stem_parent, f"{stem}.mp3")
            print(stem_parent)
            print(stem_path)
            if not os.path.exists(stem_path):
                return JSONResponse({"error": "Stem not found"})
            return FileResponse(
                stem_path,
                media_type="audio/mpeg" if format == "mp3" else "audio/wav",
            )
        else:
            # zip up the folder
            zip_filename = f"{output_folder}.zip"
            with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.basename(file_path))

            return FileResponse(
                zip_filename, filename=f"{folder_id}.zip", media_type="application/zip"
            )
    except Exception as e:
        logging.error(e, exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": "Error extracting audio"}
        )


@router.post("/demucs")
async def wav_demucs(req: WavDemucsRequest):
    return await generate(
        **req.__dict__
    )


@router.get("/demucs")
async def wav_demucs_from_url(req: WavDemucsRequest = Depends()):
    return await wav_demucs(req)
