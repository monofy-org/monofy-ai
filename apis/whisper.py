from fastapi import FastAPI


def whisper_api(app: FastAPI):
    from fastapi import WebSocket, WebSocketDisconnect
    from clients import WhisperClient

    whisper = WhisperClient()

    @app.websocket("/api/whisper/stream")
    async def whisper_stream(websocket: WebSocket):
        await websocket.accept()

        try:
            while True:
                # Receive audio chunk from the WebSocket
                chunk = await websocket.receive_bytes()

                # Process the audio chunk and get the transcription
                transcription = await whisper.process_audio_chunk(chunk)

                # Send the transcription back to the client
                await websocket.send_text(transcription)

        except WebSocketDisconnect:
            pass
