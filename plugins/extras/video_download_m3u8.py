import logging
from fastapi import HTTPException
from fastapi.responses import FileResponse
import requests
from pydantic import BaseModel

from typing import List
from modules.plugins import router
from utils.file_utils import random_filename

class DownloadM3U8Request(BaseModel):
    url: str
    
@router.post("/m3u8/download")
async def download_and_convert(req: DownloadM3U8Request):
    # Retrieve the m3u8 playlist file
    logging.info(f"Fetching m3u8 playlist: {req.url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(req.url, headers=headers)
    print(response.status_code)
    print(response.text)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch m3u8 playlist.")
    
    base_url = req.url.rsplit('/', 1)[0]

    # Parse the m3u8 content to extract .ts URLs
    ts_urls = []
    lines = response.text.splitlines()
    for line in lines:
        if line.find('.ts') > -1:
            ts_urls.append(f"{base_url}/{line}")
    
    if not ts_urls:
        logging.error("No .ts segments found in the m3u8 file.")
        raise HTTPException(status_code=400, detail="No .ts segments found in the m3u8 file.")
    
    print(ts_urls)

    # Download the .ts files    
    ts_files = []
    for ts_url in ts_urls:
        response = requests.get(ts_url, stream=True)
        if response.status_code == 200:
            ts_file = random_filename("ts")
            with open(ts_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            ts_files.append(ts_file)

    # Merge the .ts files into an MP4 file
    try:
        from ffmpy import FFmpeg
        outfile = random_filename("mp4")
        ff = FFmpeg(
                inputs={f'concat:{"|".join(ts_files)}': None},
                outputs={outfile: '-c copy'}
            )
        ff.run()
        return FileResponse(outfile, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while merging files: {str(e)}")
