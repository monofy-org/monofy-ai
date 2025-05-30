import logging
import requests
import os
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.plugins import router
from utils.file_utils import random_filename

class DownloadM3U8Request(BaseModel):
    url: str
    segment_count: int = 10  # Default to grab the last 10 segments (typically ~30-60 seconds)

@router.post("/m3u8/download")
async def download_m3u8(req: DownloadM3U8Request):
    """
    Download the most recent segments from an M3U8 stream and return as a video file.
    """
    logging.info(f"Fetching m3u8 playlist: {req.url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    
    # Get the playlist
    response = requests.get(req.url, headers=headers)
    logging.info(f"Response status: {response.status_code}")
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, detail="Failed to fetch m3u8 playlist."
        )
    
    # Extract base URL for resolving relative paths
    base_url = req.url.rsplit("/", 1)[0] + "/"
    content = response.text
    lines = content.splitlines()
    
    # Check if this is a master playlist (contains links to other m3u8 files)
    m3u8_files = [line for line in lines if line.endswith(".m3u8") and not line.startswith("#")]
    
    if m3u8_files:
        # Get the last variant (usually highest quality or most relevant)
        m3u8_file = m3u8_files[-1]
        m3u8_url = m3u8_file if m3u8_file.startswith("http") else base_url + m3u8_file
        
        logging.info(f"Using variant playlist: {m3u8_url}")
        response = requests.get(m3u8_url, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code, detail="Failed to fetch variant playlist."
            )
        
        # Update content and base URL
        content = response.text
        base_url = m3u8_url.rsplit("/", 1)[0] + "/"
        lines = content.splitlines()
    
    # Extract all .ts segment files
    ts_urls = []
    for line in lines:
        if line.endswith(".ts") and not line.startswith("#"):
            full_url = line if line.startswith("http") else base_url + line
            ts_urls.append(full_url)
    
    if not ts_urls:
        raise HTTPException(
            status_code=500, detail="No .ts files found in the m3u8 playlist."
        )
    
    # Get the most recent segments based on requested count
    recent_segments = ts_urls[-min(req.segment_count, len(ts_urls)):]
    logging.info(f"Found {len(ts_urls)} segments, using the last {len(recent_segments)}")
    
    # Download the segments
    ts_files = []
    for ts_url in recent_segments:
        try:
            logging.info(f"Downloading segment: {ts_url}")
            response = requests.get(ts_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                ts_file = random_filename("ts")
                with open(ts_file, "wb") as f:
                    f.write(response.content)
                ts_files.append(ts_file)
            else:
                logging.error(f"Failed to download {ts_url}: {response.status_code}")
        except Exception as e:
            logging.error(f"Error downloading {ts_url}: {str(e)}")
    
    if not ts_files:
        raise HTTPException(
            status_code=500, detail="No segments downloaded successfully."
        )
    
    # Merge the segments into a single file
    try:
        from ffmpy import FFmpeg
        outfile = random_filename("mp4")
        
        # Use concat demuxer for better compatibility
        concat_list = "concat:" + "|".join(ts_files)
        ff = FFmpeg(
            inputs={concat_list: None},
            outputs={outfile: "-c copy"}
        )
        
        logging.info(f"Running FFmpeg to merge {len(ts_files)} segments")
        ff.run()
        
        # Clean up temporary files
        for ts_file in ts_files:
            try:
                os.remove(ts_file)
            except:
                pass
        
        return FileResponse(outfile, media_type="video/mp4")
    except Exception as e:
        # Clean up on error
        for ts_file in ts_files:
            try:
                os.remove(ts_file)
            except:
                pass
        
        logging.error(f"Error merging segments: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error merging segments: {str(e)}"
        )