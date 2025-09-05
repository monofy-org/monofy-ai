import logging
import requests
import os
import re
from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.plugins import router
from utils.file_utils import random_filename


class DownloadM3U8Request(BaseModel):
    url: str
    segment_count: int = (
        10  # Default to grab the last 10 segments (typically ~30-60 seconds)
    )


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

    # --- Start of the corrected logic for master playlists ---
    m3u8_streams = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#EXT-X-STREAM-INF:"):
            try:
                # Use a more flexible regex to handle both quoted and unquoted values
                attributes_str = line.split(":", 1)[1]
                attributes = dict(re.findall(r'(\w+)=(".*?"|[^,"]+)', attributes_str))
                
                # Find the next non-comment, non-empty line
                playlist_url_found = False
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line and not next_line.startswith("#"):
                        playlist_url = next_line
                        playlist_url_found = True
                        break
                    j += 1
                
                if playlist_url_found:
                    # Use bandwidth to select the best quality
                    bandwidth = int(attributes.get("BANDWIDTH", 0))
                    m3u8_streams[bandwidth] = playlist_url
                    i = j  # Move the main index forward to the URL's position
                else:
                    logging.warning(f"Skipping #EXT-X-STREAM-INF tag with no playlist URL: {line}")
                    i += 1
            except (ValueError, KeyError, IndexError):
                logging.warning(f"Skipping malformed #EXT-X-STREAM-INF tag: {line}")
                i += 1
        else:
            i += 1
    
    if m3u8_streams:
        # Find the variant with the highest bandwidth
        highest_bandwidth = max(m3u8_streams.keys())
        m3u8_file = m3u8_streams[highest_bandwidth]

        m3u8_url = m3u8_file if m3u8_file.startswith("http") else base_url + m3u8_file

        logging.info(f"Using highest quality variant playlist: {m3u8_url}")
        response = requests.get(m3u8_url, headers=headers)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to fetch variant playlist.",
            )

        # Update content and base URL
        content = response.text
        base_url = m3u8_url.rsplit("/", 1)[0] + "/"
        lines = content.splitlines()    

    # Extract all .ts or .m4s segment files
    ts_urls = []
    for line in lines:
        # Check if the line is a segment URI, which typically does not start with #
        if not line.startswith("#"):
            # Check if the line contains a known segment extension
            if ".ts" in line or ".m4s" in line:
                full_url = line if line.startswith("http") else base_url + line
                ts_urls.append(full_url)

    if not ts_urls:
        raise HTTPException(
            status_code=500, detail="No .ts or .m4s segment files found in the playlist."
        )

    # Get the most recent segments based on requested count
    recent_segments = ts_urls[-min(req.segment_count, len(ts_urls)) :]
    logging.info(
        f"Found {len(ts_urls)} segments, using the last {len(recent_segments)}"
    )

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
        
        # Find the init segment URL
        init_segment_url = None
        for line in lines:
            if line.startswith("#EXT-X-MAP:"):
                # Use regex to extract the URI from the tag
                match = re.search(r'URI="(.*?)"', line)
                if match:
                    init_segment_url = match.group(1)
                    # Resolve to a full URL
                    if not init_segment_url.startswith("http"):
                        init_segment_url = base_url + init_segment_url
                    break
        
        # Download the init segment if found
        init_segment_file = None
        if init_segment_url:
            logging.info(f"Downloading initialization segment: {init_segment_url}")
            try:
                response = requests.get(init_segment_url, headers=headers, timeout=15)
                if response.status_code == 200:
                    init_segment_file = random_filename("mp4")
                    with open(init_segment_file, "wb") as f:
                        f.write(response.content)
                    logging.info("Initialization segment downloaded successfully.")
                else:
                    logging.error(f"Failed to download init segment {init_segment_url}: {response.status_code}")
            except Exception as e:
                logging.error(f"Error downloading init segment: {str(e)}")

        # Build the concat list, starting with the init segment if it exists
        if init_segment_file:
            concat_list = "concat:" + init_segment_file + "|" + "|".join(ts_files)
        else:
            concat_list = "concat:" + "|".join(ts_files)

        ff = FFmpeg(inputs={concat_list: None}, outputs={outfile: "-c copy"})

        logging.info(f"Running FFmpeg to merge segments")
        ff.run()

        # Clean up temporary files
        for ts_file in ts_files:
            try:
                os.remove(ts_file)
            except:
                pass
        if init_segment_file:
            try:
                os.remove(init_segment_file)
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
        if init_segment_file:
            try:
                os.remove(init_segment_file)
            except:
                pass

        logging.error(f"Error merging segments: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error merging segments: {str(e)}")