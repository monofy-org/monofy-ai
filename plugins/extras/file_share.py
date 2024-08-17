import os
import shutil
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse
from modules.plugins import router
from utils.file_utils import ensure_folder_exists, random_filename

UPLOAD_DIR = ".cache/user-shared"


@router.post("/file_share")
async def file_share(file: UploadFile = File(...)):

    # make sure file is a wav or mp3 or video
    if file.content_type not in [
        "audio/wav",
        "audio/mpeg",
        "video/mp4",
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "application/x-javascript",
        "text/css"
    ]:
        return {"error": "Unsupported media: " + file.content_type }

    ensure_folder_exists(UPLOAD_DIR)

    base_filename = random_filename(file.filename.split(".")[-1], False)

    filename = os.path.join(UPLOAD_DIR, base_filename)
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return JSONResponse({"filename": base_filename})
