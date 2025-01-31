import os
import shlex
import shutil
import uuid
import subprocess
import tempfile

from fastapi.responses import FileResponse, JSONResponse
from modules.plugins import PluginBase, use_plugin_unsafe
from pathlib import Path
from pydantic import BaseModel

from plugins.exllamav2 import ExllamaV2Plugin


class VideoAIRequest(BaseModel):
    prompt: str
    files: list[str]    


class VideoAIPlugin(PluginBase):
    name = "Video AI"
    description = "Edit videos using AI and FFMpeg"
    instance = None
    plugins = ["ExllamaV2Plugin"]

    def __init__(self):
        self.model = None
        pass

    def generate(self, prompt, files):
        plugin: ExllamaV2Plugin = None
        try:
            plugin = use_plugin_unsafe(ExllamaV2Plugin)
            filename, command = update(plugin, files, prompt)
            print(command)
            return FileResponse(filename, filename=os.path.basename(filename))
        except Exception as e:
            print(e)
            return JSONResponse({"error": str(e)}, status_code=500)


def get_files_infos(files):
    from PIL import Image
    from moviepy.editor import VideoFileClip, AudioFileClip

    results = []
    for file in files:
        file_path = Path(file.name)
        info = {}
        info["size"] = os.path.getsize(file_path)
        # Sanitize filename by replacing spaces with underscores
        info["name"] = file_path.name.replace(" ", "_")
        file_extension = file_path.suffix

        if file_extension in (".mp4", ".avi", ".mkv", ".mov"):
            info["type"] = "video"
            video = VideoFileClip(file.name)
            info["duration"] = video.duration
            info["dimensions"] = "{}x{}".format(video.size[0], video.size[1])
            if video.audio:
                info["type"] = "video/audio"
                info["audio_channels"] = video.audio.nchannels
            video.close()
        elif file_extension in (".mp3", ".wav"):
            info["type"] = "audio"
            audio = AudioFileClip(file.name)
            info["duration"] = audio.duration
            info["audio_channels"] = audio.nchannels
            audio.close()
        elif file_extension in (
            ".png",
            ".jpg",
            ".jpeg",
            ".tiff",
            ".bmp",
            ".gif",
            ".svg",
        ):
            info["type"] = "image"
            img = Image.open(file.name)
            info["dimensions"] = "{}x{}".format(img.size[0], img.size[1])
        results.append(info)
    return results


def get_completion(plugin: ExllamaV2Plugin, prompt, files_info, top_p, temperature):
    # Create table header
    files_info_string = "| Type | Name | Dimensions | Duration | Audio Channels |\n"
    files_info_string += "|------|------|------------|-----------|--------|\n"

    # Add each file as a table row
    for file_info in files_info:
        dimensions = file_info.get("dimensions", "-")
        duration = (
            f"{file_info.get('duration', '-')}s" if "duration" in file_info else "-"
        )
        audio = (
            f"{file_info.get('audio_channels', '-')} channels"
            if "audio_channels" in file_info
            else "-"
        )

        files_info_string += f"| {file_info['type']} | {file_info['name']} | {dimensions} | {duration} | {audio} |\n"

    messages = [
        {
            "role": "system",
            "content": """
You are a very experienced media engineer, controlling a UNIX terminal.
You are an FFMPEG expert with years of experience and multiple contributions to the FFMPEG project.
You are given:
(1) a set of video, audio and/or image assets. Including their name, duration, dimensions and file size
(2) the description of a new video you need to create from the list of assets
Your objective is to generate the SIMPLEST POSSIBLE single ffmpeg command to create the requested video.
Key requirements:
    - Use the absolute minimum number of ffmpeg options needed
    - Avoid complex filter chains or filter_complex if possible
    - Prefer simple concatenation, scaling, and basic filters
    - Output exactly ONE command that will be directly pasted into the terminal
    - Never output multiple commands chained together
    - Output the command in a single line (no line breaks or multiple lines)
    - If the user asks for waveform visualization make sure to set the mode to `line` with and the use the full width of the video. Also concatenate the audio into a single channel.
    - For image sequences: Use -framerate and pattern matching (like 'img%d.jpg') when possible, falling back to individual image processing with -loop 1 and appropriate filters only when necessary.
    - When showing file operations or commands, always use explicit paths and filenames without wildcards - avoid using asterisk (*) or glob patterns. Instead, use specific numbered sequences (like %d), explicit file lists, or show the full filename.
Remember: Simpler is better. Only use advanced ffmpeg features if absolutely necessary for the requested output.
""",
        },
        {
            "role": "user",
            "content": f"""Always output the media as video/mp4 and output file with "output.mp4". Provide only the shell command without any explanations.
The current assets and objective follow. Reply with the FFMPEG command:
AVAILABLE ASSETS LIST:
{files_info_string}
OBJECTIVE: {prompt} and output at "output.mp4"
YOUR FFMPEG COMMAND:
         """,
        },
    ]
    try:
        # Print the complete prompt
        print("\n=== COMPLETE PROMPT ===")
        for msg in messages:
            print(f"\n[{msg['role'].upper()}]:")
            print(msg["content"])
        print("=====================\n")

        completion = plugin.generate_chat_response(            
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=2048,
        )
        content = completion.choices[0].message.content
        # Extract command from code block if present
        if "```" in content:
            # Find content between ```sh or ```bash and the next ```
            import re

            command = re.search(r"```(?:sh|bash)?\n(.*?)\n```", content, re.DOTALL)
            if command:
                command = command.group(1).strip()
            else:
                command = content.replace("\n", "")
        else:
            command = content.replace("\n", "")

        # remove output.mp4 with the actual output file path
        command = command.replace("output.mp4", "")

        return command
    except Exception as e:
        raise Exception("API Error")


def update(
    plugin: ExllamaV2Plugin, files: list, prompt: str, top_p: float = 1, temperature=1
):
    if prompt == "":
        raise Exception("Please enter a prompt.")

    files_info = get_files_infos(files)
    # disable this if you're running the app locally or on your own server
    for file_info in files_info:
        if file_info["type"] == "video":
            if file_info["duration"] > 120:
                raise Exception(
                    "Please make sure all videos are less than 2 minute long."
                )
        if file_info["size"] > 10000000:
            raise Exception("Please make sure all files are less than 10MB in size.")

    attempts = 0
    while attempts < 2:
        print("ATTEMPT", attempts)
        try:
            command_string = get_completion(plugin, prompt, files_info, top_p, temperature)
            print(
                f"""///PROMTP {prompt} \n\n/// START OF COMMAND ///:\n\n{command_string}\n\n/// END OF COMMAND ///\n\n"""
            )

            # split command string into list of arguments
            args = shlex.split(command_string)
            if args[0] != "ffmpeg":
                raise Exception("Command does not start with ffmpeg")
            temp_dir = tempfile.mkdtemp()
            # copy files to temp dir with sanitized names
            for file in files:
                file_path = Path(file.name)
                sanitized_name = file_path.name.replace(" ", "_")
                shutil.copy(file_path, Path(temp_dir) / sanitized_name)

            # test if ffmpeg command is valid dry run
            ffmpg_dry_run = subprocess.run(
                args + ["-f", "null", "-"],
                stderr=subprocess.PIPE,
                text=True,
                cwd=temp_dir,
            )
            if ffmpg_dry_run.returncode == 0:
                print("Command is valid.")
            else:
                print("Command is not valid. Error output:")
                print(ffmpg_dry_run.stderr)
                raise Exception(
                    "FFMPEG generated command is not valid. Please try something else."
                )

            output_file_name = f"output_{uuid.uuid4()}.mp4"
            output_file_path = str((Path(temp_dir) / output_file_name).resolve())
            final_command = args + ["-y", output_file_path]
            print(
                f"\n=== EXECUTING FFMPEG COMMAND ===\nffmpeg {' '.join(final_command[1:])}\n"
            )
            subprocess.run(final_command, cwd=temp_dir)
            generated_command = f"### Generated Command\n```bash\nffmpeg {' '.join(args[1:])} -y output.mp4\n```"
            return output_file_path, generated_command
        except Exception as e:
            attempts += 1
            if attempts >= 2:
                print("FROM UPDATE", e)
                raise e
