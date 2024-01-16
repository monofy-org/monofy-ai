from fastapi import UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

router = APIRouter()


@router.websocket("/whisper/stream")
async def whisper_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        from clients import WhisperClient

        while True:
            chunk = await websocket.receive_bytes()
            transcription = WhisperClient.process_audio_chunk(chunk)
            await websocket.send_text(transcription)

    except WebSocketDisconnect:
        pass


@router.post("/whisper")
async def whisper(file: UploadFile = File(...)):
    from clients import WhisperClient

    contents = await file.read()
    transcription = WhisperClient.process_audio_file(contents)

    return {"transcription": transcription}
