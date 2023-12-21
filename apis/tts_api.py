import logging
import time
from fastapi import FastAPI, Query, WebSocket, HTTPException
from fastapi.responses import Response, JSONResponse, FileResponse
from ttsclient import TTSClient
from settings import LOG_LEVEL
import edge_tts
from edge_tts import VoicesManager

logging.basicConfig(level=LOG_LEVEL)

edge_voices: VoicesManager = None
edge_voice = None


async def edge_initialize():
    global edge_voice, edge_voices
    if edge_voices is None:
        edge_voices = await VoicesManager.create()
        edge_voice = edge_voices.find(Gender="Male", Language="es")
        print("Got edge voices:")
        print(edge_voices)


def tts_api(app: FastAPI):
    tts = TTSClient.instance

    @app.get("/api/tts/voices", response_model=dict)
    async def voice_list():
        voices = await tts.list_voices()
        if voices is None:
            return JSONResponse(
                content={"error": "Error retrieving voice list"}, status_code=500
            )
        return voices

    @app.websocket("/api/tts/stream")
    async def api_tts_stream(
        websocket: WebSocket,
        text: str = Query(
            ..., title="Text", description="The text to convert to speech"
        ),
        voice: str = Query("female1", title="Voice", description="The voice to use"),
        speed: int = Query(
            1,
            title="Speed",
            description="Speed (default = 1.0)",
        ),
        temperature: int = Query(
            0.75,
            title="Temperature",
            description="Temperature (default = 0.75)",
        ),
        emotion: str = Query(
            "Neutral",
            title="Emotion",
            description="Emotion to use (default = Neutral)",
        ),
        language: str = Query(
            "en",
            title="Language",
            description="Language to use (default = en)",
        ),
        model: str = Query(
            "xtts",
            title="TTS Model",
            description="TTS Model to use (default = xtts)",
        ),
    ):
        await websocket.accept()

        if model == "edge-tts":
            if edge_voices is None:
                await edge_initialize()
            print("EDGE VOICE = " + edge_voice)
            communicate = edge_tts.Communicate(text, edge_voice["Name"])
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    print("DEBUG:")
                    print(chunk)

        else:
            tts.load_speaker(f"voices/{voice}.wav")

            async for chunk in tts.generate_speech_streaming(
                text=text,
                speed=speed,
                temperature=temperature,
                emotion=emotion,
                language=language,
                speaker_wav=f"voices/{voice}.wav",
            ):
                await websocket.send_bytes(chunk)

        try:
            await websocket.close()
        except Exception:
            pass

    @app.get("/api/tts", response_model=dict)
    async def text_to_speech(
        text: str = Query(
            ..., title="Text", description="The text to convert to speech"
        ),
        voice: str = Query("female1", title="Voice", description="The voice to use"),
        speed: int = Query(
            1,
            title="Speed",
            description="Speed (default = 1.0)",
        ),
        temperature: int = Query(
            0.75,
            title="Temperature",
            description="Temperature (default = 0.75)",
        ),
        emotion: str = Query(
            "Neutral",
            title="Emotion",
            description="Emotion to use (default = Neutral)",
        ),
        language: str = Query(
            "en",
            title="Language",
            description="Language to use (default = en)",
        ),
        model: str = Query(
            "xttsv2",
            title="TTS Model",
            description="TTS Model to use (default = xttsv2)",
        ),
    ):
        try:
            if model == "edge-tts":
                communicate = edge_tts.Communicate(text=text)
                path = f".cache/{time.time()}.mp3"
                await communicate.save(path)

                # Create a FileResponse
                response = FileResponse(path=path, media_type="audio/mpeg")

                # Delete the file after sending the response
                # os.remove(path)

                return response

            else:
                wav_bytes = tts.generate_speech(
                    text=text,
                    speed=speed,
                    speaker_wav=f"voices/{voice}.wav",
                    temperature=temperature,
                    emotion=emotion,
                    language=language,
                )
                return Response(content=wav_bytes, media_type="audio/wav")

        except Exception as e:
            return HTTPException(500, f"An error occurred: {str(e)}")
