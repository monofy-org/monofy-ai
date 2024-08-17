import logging
import os
from typing import Literal
import uuid
import zipfile
import demucs.separate
from fastapi import Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from modules.plugins import router
from utils.audio_utils import get_audio_from_request
from utils.file_utils import random_filename


class WavDemucsRequest(BaseModel):
    url: str
    stem: Literal["vocals", "drums", "bass", "other", None] = None
    model: Literal["htdemucs", "mdx"] = "mdx"


@router.post("/demucs")
async def wav_demucs(req: WavDemucsRequest):
    try:
        extension = req.url.split(".")[-1]
        audio_path = random_filename(extension)
        audio_path = get_audio_from_request(req.url, audio_path)
        folder_id = str(uuid.uuid4())
        random_name = os.path.abspath(os.path.join(".cache", folder_id))
        os.mkdir(random_name)

        demucs.separate.main(
            ["--two-stems", req.stem, "-n", req.model, "-o", random_name, audio_path]            
            if req.stem
            else ["-n", req.model, "-o", random_name, audio_path]            
        )

        # zip up the folder
        zip_filename = f"{random_name}.zip"
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(random_name):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.basename(file_path))

        return FileResponse(
            zip_filename, filename=f"{folder_id}.zip", media_type="application/zip"
        )
    except Exception as e:
        logging.error(e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": f"Error: {e}"},
        )


@router.get("/demucs")
async def wav_demucs_from_url(req: WavDemucsRequest = Depends()):
    return await wav_demucs(req)
