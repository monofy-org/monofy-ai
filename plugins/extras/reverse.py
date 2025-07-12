import io
import os

import soundfile as sf
from fastapi.responses import FileResponse
from moviepy import VideoFileClip
from pydantic import BaseModel

from modules import plugins
from utils.file_utils import random_filename
from utils.video_utils import get_video_from_request


class ReverseRequest(BaseModel):
    video: str


def reverse_video(video_path, output_path):
    """
    Reverses the video at the given path and saves it to the given output path.
    """
    # Load the video
    video: VideoFileClip = VideoFileClip(video_path)
    # Reverse the video
    reversed_frames = list(reversed(list(video.iter_frames())))
    reversed_video = VideoFileClip(video_path)
    reversed_video.frame_function = lambda t: reversed_frames[int(t * video.fps)]

    audio = video.audio
    reversed_audio = io.BytesIO()
    sf.write(
        reversed_audio, audio.to_soundarray()[::-1], audio.fps, format="wav"
    )
    reversed_audio.seek(0)

    # Save the reversed video
    reversed_video.write_videofile(output_path, fps=video.fps, audio=reversed_audio)

    # Clean up
    video.close()
    reversed_video.close()

    return output_path


@plugins.router.post("/reverse")
async def reverse(req: ReverseRequest):
    input_path = get_video_from_request(req.video)
    output_path = random_filename("mp4")
    reverse_video(input_path, output_path)
    return FileResponse(output_path, filename=os.path.basename(output_path), media_type="video/mp4")
    

