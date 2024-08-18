import logging
import os
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from modules.plugins import router
from utils.audio_utils import get_audio_from_request
from utils.file_utils import ensure_folder_exists, random_filename

checkpoint_path = os.path.join("models", "piano2midi")

class Piano2MidiRequest(BaseModel):
    audio: str


@router.post("/piano2midi")
async def piano2mid(req: Piano2MidiRequest):

    model_path = os.path.join(checkpoint_path, "CRNN_note_F1=0.9677_pedal_F1=0.9186.pth")

    if not os.path.exists(model_path):
        logging.info(f"Downloading model to {model_path}")
        import requests
        url = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
        response = requests.get(url)
        ensure_folder_exists(checkpoint_path)
        with open(model_path, "wb") as file:
            file.write(response.content)

    audio_path = random_filename("mp3")
    audio_path = get_audio_from_request(req.audio, audio_path)
    from piano_transcription_inference import (
        PianoTranscription,
        sample_rate,
        load_audio,
    )

    audio, _ = load_audio(audio_path, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(
        device="cuda", checkpoint_path=model_path
    )  # device: 'cuda' | 'cpu'
    filename = random_filename("mid")
    transcribed_dict = transcriptor.transcribe(audio, filename)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    if os.path.exists(filename):        
        return FileResponse(filename, filename=os.path.basename(filename))
    else:
        raise HTTPException(status_code=500, detail="Failed to generate MIDI file")


@router.get("/piano2midi")
async def piano2mid_from_url(
    req: Piano2MidiRequest = Depends(),
):
    return await piano2mid(req)
