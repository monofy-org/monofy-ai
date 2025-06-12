import logging
import os
from typing import Optional
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from modules.plugins import router
from plugins.extras import wav_demucs
from utils.audio_utils import get_audio_from_request
from utils.file_utils import ensure_folder_exists, random_filename

checkpoint_folder = os.path.join("models", "piano2midi")
checkpoint_path = os.path.join(
    checkpoint_folder, "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class Piano2MidiRequest(BaseModel):
    audio: str
    isolate: Optional[bool] = False


def ensure_model_exists():
    if not os.path.exists(checkpoint_path):
        logging.info(f"Downloading model to {checkpoint_folder}")
        import requests

        url = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
        response = requests.get(url)
        ensure_folder_exists(checkpoint_folder)
        with open(checkpoint_path, "wb") as file:
            file.write(response.content)


@router.post("/piano2midi", tags=["Audio and Music"])
async def piano2mid(req: Piano2MidiRequest):

    from submodules.piano_transcription_inference.piano_transcription_inference import (
        PianoTranscription,
        sample_rate,
        load_audio,
    )

    ensure_model_exists()

    audio_path = get_audio_from_request(req.audio)

    print(audio_path)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="Invalid audio file")

    if req.isolate:
        logging.info("Isolating audio...")
        response = await wav_demucs.generate(url=audio_path, stem="other", stem_only=True)
        if not hasattr(response, "path"):
            raise HTTPException(status_code=500, detail="Failed to isolate audio")
        audio_path = response.path

    logging.info("Transcribing audio...")
    audio, _ = load_audio(audio_path, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(
        device=device, checkpoint_path=checkpoint_path
    )  # device: 'cuda' | 'cpu'
    filename = random_filename("mid")
    transcribed_dict = transcriptor.transcribe(audio, filename)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    if os.path.exists(filename):
        return FileResponse(filename, filename=os.path.basename(filename))
    else:
        raise HTTPException(status_code=500, detail="Failed to generate MIDI file")


@router.get("/piano2midi", tags=["Audio and Music"])
async def piano2mid_from_url(
    req: Piano2MidiRequest = Depends(),
):
    return await piano2mid(req)
