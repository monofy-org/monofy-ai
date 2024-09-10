import logging
import os
from typing import Literal, Optional
import uuid
import zipfile
import demucs.separate
from fastapi import Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from modules.plugins import router
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

def generate(**kwargs):
    audio: str = kwargs.get("url")
    stem: str = kwargs.get("stem")
    model: str = kwargs.get("model", "mdx_extra")
    format: str = kwargs.get("format", "mp3")
    mp3_bitrate: int = kwargs.get("mp3_bitrate", 192)
    stem_only: bool = kwargs.get("stem_only", False)

    try:            
        audio_path = get_audio_from_request(audio)
        folder_id = str(uuid.uuid4())
        random_name = os.path.abspath(os.path.join(CACHE_PATH, folder_id))
        os.mkdir(random_name)

        args = []
        if format == "mp3":
            args.extend(["--mp3", "--mp3-bitrate", str(mp3_bitrate)])
        elif format != "wav":
            format = "wav"
        if stem:
            args.extend(["--two-stems", stem])
        args.extend(["-n", model, "-o", random_name, audio_path])

        demucs.separate.main(args)

        if stem_only:
            stem_parent = stem_path = os.path.join(random_name, model)            
            stem_parent  = os.path.join(stem_parent, os.listdir(stem_parent)[0])
            stem_path = os.path.join(stem_parent, f"{stem}.mp3")
            print(stem_parent)
            print(stem_path)
            if not os.path.exists(stem_path):
                return JSONResponse({"error": "Stem not found"})
            return FileResponse(
                stem_path,
                media_type="audio/mpeg" if format == "mp3" else "audio/wav",
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
    
    

@router.post("/demucs")
async def wav_demucs(req: WavDemucsRequest):
    return generate(
        url=req.url,
        stem=req.stem,
        model=req.model,
        format=req.format,
        mp3_bitrate=req.mp3_bitrate,
        stem_only=req.stem_only,
    )


@router.get("/demucs")
async def wav_demucs_from_url(req: WavDemucsRequest = Depends()):
    return await wav_demucs(req)
