import logging
import os
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from modules.plugins import router
from utils.file_utils import random_filename
from utils.video_utils import get_video_from_request
from ffmpy import FFmpeg
import tempfile

class VideoCombineRequest(BaseModel):
    videos: list[str]
    crossfade: float = 0.0  # seconds

@router.post("/video/combine")
async def combine_videos(req: VideoCombineRequest) -> dict:
    """Combine multiple video files into one using ffmpeg."""

    files = [get_video_from_request(video) for video in req.videos]
    if not files:
        return JSONResponse(status_code=400, content={"error": "No valid video files provided."})
    
    try:
        from moviepy import VideoFileClip, concatenate_videoclips

        clips = [VideoFileClip(file) for file in files]

        if req.crossfade > 0:
            # Use ffmpeg via ffmpy to concatenate with crossfade

            # Prepare input files list for ffmpeg
            temp_files = []
            for i, clip in enumerate(clips):
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                clip.write_videofile(temp_file.name, codec="libx264")
                temp_files.append(temp_file.name)
                temp_file.close()

            # Build ffmpeg filter_complex for crossfading
            filter_complex = ""
            inputs = ""
            for _, file in enumerate(temp_files):
                inputs += f"-i {file} "
            # Build filter_complex for chaining xfade and acrossfade filters
            n = len(temp_files)
            fade = req.crossfade
            filter_steps = []
            # Video xfade chain
            for i in range(n - 1):
                if i == 0:
                    v_in1 = f"[0:v]"
                    v_in2 = f"[1:v]"
                    v_out = f"[v1]"
                    offset = clips[0].duration - fade
                else:
                    v_in1 = f"[v{i}]"
                    v_in2 = f"[{i+1}:v]"
                    v_out = f"[v{i+1}]"
                    offset = sum(c.duration for c in clips[:i+1]) - fade
                filter_steps.append(
                    f"{v_in1}{v_in2}xfade=transition=fade:duration={fade}:offset={offset}{v_out}"
                )
            # Audio acrossfade chain
            for i in range(n - 1):
                if i == 0:
                    a_in1 = f"[0:a]"
                    a_in2 = f"[1:a]"
                    a_out = f"[a1]"
                else:
                    a_in1 = f"[a{i}]"
                    a_in2 = f"[{i+1}:a]"
                    a_out = f"[a{i+1}]"
                filter_steps.append(
                    f"{a_in1}{a_in2}acrossfade=d={fade}{a_out}"
                )
            last_v = f"v{n-1}"
            last_a = f"a{n-1}"
            filter_complex = ";".join(filter_steps)
            output = random_filename("mp4")
            ff = FFmpeg(
                inputs={file: None for file in temp_files},
                outputs={output: [
                    '-filter_complex', filter_complex,
                    '-map', f'[{last_v}]',
                    '-map', f'[{last_a}]',
                    '-c:v', 'libx264', '-c:a', 'aac', '-y'
                ]}
            )
            ff.run()
            final_clip = VideoFileClip(output)
            # Clean up temp files
            for f in temp_files:
                os.unlink(f)
        else:
            final_clip = concatenate_videoclips(clips)

        output = random_filename("mp4")
        final_clip.write_videofile(output, codec="libx264")
        for clip in clips:
            clip.close()
        final_clip.close()
        return FileResponse(output, media_type="video/mp4", filename=output)
    except Exception as e:
        logging.error(f"Error combining videos: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})