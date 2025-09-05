import os
from fastapi.responses import FileResponse, JSONResponse
import requests
import json
from modules.plugins import router
from plugins.extras.video_download_m3u8 import DownloadM3U8Request, download_m3u8
from settings import CACHE_PATH

master_playlist = "https://iptv-org.github.io/iptv/index.m3u"
cache_file = os.path.join("public_html", "iptv", "streams.json")

os.makedirs(CACHE_PATH, exist_ok=True)

def parse_m3u_to_json(m3u_content):
    lines = m3u_content.splitlines()
    results = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#EXTINF:"):
            # Find the next non-empty line (should be the stream URL)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                url = lines[j].strip()
                name = line.split(",")[-1]
                logo = ""
                group = ""
                if 'tvg-logo="' in line:
                    logo = line.split('tvg-logo="')[1].split('"')[0]
                if 'group-title="' in line:
                    group = line.split('group-title="')[1].split('"')[0]
                results.append({"name": name, "url": url, "logo": logo, "group": group})
            i = j
        else:
            i += 1
    return results


if not os.path.exists(cache_file):
    response = requests.get(master_playlist)
    response.raise_for_status()
    channels = parse_m3u_to_json(response.text)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(channels, f, ensure_ascii=False, indent=2)


@router.get("/iptv/streams.json")
def iptv_streams():
    return FileResponse(cache_file, media_type="application/json", filename="streams.json")

@router.get("/iptv/favorites.txt")
def iptv_streams():
    return FileResponse(cache_file, media_type="text/plain", filename="favorites.txt")


@router.get("/iptv/search")
def iptv_search(query: str):
    with open(cache_file, "r", encoding="utf-8") as f:
        channels = json.load(f)
    results = [
        channel for channel in channels if query.lower() in channel["name"].lower()
    ]
    return {"results": results}

@router.get("/iptv/preview/{channel_name}")
async def iptv_preview(channel_name: str):

    # search for the channel in the cached JSON file
    with open(cache_file, "r", encoding="utf-8") as f:
        channels = json.load(f)
    channel = next((ch for ch in channels if ch["name"].lower() == channel_name.lower()), None)
    if not channel:
        return {"error": "Channel not found"}
    stream_url = channel["url"]

    return await download_m3u8(DownloadM3U8Request(url=stream_url, segment_count=3))
