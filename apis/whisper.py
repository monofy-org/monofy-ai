from fastapi import UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

from utils.audio_utils import resample_wav

router = APIRouter()


@router.websocket("/whisper/stream")
async def whisper_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        from clients import WhisperClient

        while True:
            # Receive audio chunk from the WebSocket
            chunk = await websocket.receive_bytes()

            # Process the audio chunk and get the transcription
            transcription = await WhisperClient.process_audio_chunk(chunk)

            # Send the transcription back to the client
            await websocket.send_text(transcription)

    except WebSocketDisconnect:
        pass


@router.post("/whisper")
async def process_wav_file(file: UploadFile = File(...)):
    from clients import WhisperClient

    # Read the uploaded WAV file
    contents = await file.read()
    converted = resample_wav(contents, 16_000)

    # Process the audio and get the transcription
    transcription = await WhisperClient.process_audio_chunk(converted)

    return {"transcription": transcription}

