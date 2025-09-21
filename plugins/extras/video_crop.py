import logging
import os
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from moviepy import VideoFileClip
from modules.plugins import router
from settings import CACHE_PATH
from utils.file_utils import random_filename
from utils.video_utils import get_video_from_request


class VideoCropRequest(BaseModel):
    video: str
    width: int = Field(default=640, gt=0, le=7680)  # Max 8K resolution width
    height: int = Field(default=480, gt=0, le=4320)  # Max 8K resolution height

async def crop_video(req: VideoCropRequest) -> VideoFileClip:
    from moviepy.video.fx.crop import crop
    from moviepy.video.fx.resize import resize

    # Load the video file
    video_path = get_video_from_request(req.video)
    video = VideoFileClip(video_path)

    # Calculate crop dimensions to center the video
    x_center = video.w / 2
    crop_x1 = x_center - req.width / 2
    crop_x2 = x_center + req.width / 2

    # Crop the video
    cropped = crop(video, x1=crop_x1, y1=0, x2=crop_x2, y2=req.height)

    # Resize if needed to ensure exact dimensions
    final = resize(cropped, width=req.width, height=req.height)

    return final


@router.post("/video/crop")
async def video_crop_and_resize(request: VideoCropRequest):
    try:
        clip: VideoFileClip = None

        filename = random_filename("mp4", include_cache_path=False)
        output_path = os.path.join(CACHE_PATH, filename)

        logging.info(f"Cropping video: {request.video} to {request.width}x{request.height}")

        clip = await crop_video(request)
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        clip.close()

        return StreamingResponse(FileResponse(output_path), media_type="video/mp4")

    except Exception as e:
        logging.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if clip:
            clip.close()
