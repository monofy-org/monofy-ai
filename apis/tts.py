import logging
import os
import time
from fastapi import Query, WebSocket, HTTPException
from fastapi.routing import APIRouter
from fastapi.responses import Response, JSONResponse, FileResponse
import edge_tts
from edge_tts import VoicesManager
from settings import TTS_VOICES_PATH

edge_voices: VoicesManager = None
edge_voice = None

router = APIRouter()


async def edge_initialize():
    global edge_voice, edge_voices
    if edge_voices is None:
        edge_voices = await VoicesManager.create()
        edge_voice = edge_voices.find(Gender="Male", Language="es")
        print("Got edge voices:")
        print(edge_voices)


@router.websocket("/tts/stream")
async def tts_stream(
    websocket: WebSocket,
    text: str = Query(..., title="Text", description="The text to convert to speech"),
    voice: str = Query("female1", title="Voice", description="The voice to use"),
    speed: int = Query(
        1,
        title="Speed",
        description="Speed (default = 1.0)",
    ),
    temperature: float = Query(
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
        from clients import TTSClient

        TTSClient.load_speaker(os.path.join(TTS_VOICES_PATH, f"{voice}.wav"))

        async for chunk in TTSClient.generate_speech_streaming(
            text=text,
            speed=speed,
            temperature=temperature,
            emotion=emotion,
            language=language,
            speaker_wav=os.path.join(TTS_VOICES_PATH, f"{voice}.wav"),
        ):
            await websocket.send_bytes(chunk)

    try:
        await websocket.close()
    except Exception as e:
        logging.error(e)
        pass


@router.get("/tts", response_model=dict)
async def text_to_speech(
    text: str = Query(..., title="Text", description="The text to convert to speech"),
    voice: str = Query("female1", title="Voice", description="The voice to use"),
    speed: int = Query(
        1,
        title="Speed",
        description="Speed (default = 1.0)",
    ),
    temperature: float = Query(
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
            from clients import TTSClient

            wav_bytes = TTSClient.generate_speech(
                text=text,
                speed=speed,
                speaker_wav=os.path.join(TTS_VOICES_PATH, f"{voice}.wav"),
                temperature=temperature,
                emotion=emotion,
                language=language,
            )
            return Response(content=wav_bytes, media_type="audio/wav")

    except Exception as e:
        logging.error(e)
        return HTTPException(500, f"An error occurred: {str(e)}")


@router.get("/tts/voices", response_model=dict)
async def voice_list():
    from clients import TTSClient

    voices = await TTSClient.list_voices()
    if voices is None:
        return JSONResponse(
            content={"error": "Error retrieving voice list"}, status_code=500
        )
    return voices
