import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import os
import librosa
import numpy as np
import requests
from hashlib import sha256
from settings import MEDIA_CACHE_DIR
import soundfile as sf

logging.basicConfig(level=logging.INFO)

def split_api(app: FastAPI):

  def download_file(url, cache_folder):
      # Generate a hash of the URL as the filename
      hash = sha256(url.encode()).hexdigest()
      filename = hash + "." + url.split(".")[-1]
      file_path = os.path.join(cache_folder, filename)

      # Check if the file already exists in the cache
      if os.path.exists(file_path):
          return file_path

      # Download the file from the URL
      response = requests.get(url, stream=True)
      with open(file_path, 'wb') as file:
          for chunk in response.iter_content(chunk_size=8192):
              file.write(chunk)

      return file_path

  def separate_vocals_and_save(name, file_path):
      # Create a folder for cache if it doesn't exist
      cache_folder = MEDIA_CACHE_DIR
      os.makedirs(cache_folder, exist_ok=True)

      # Read the audio file
      y, sr = librosa.load(file_path)

      # Decompose the spectrogram
      S_full, phase = librosa.magphase(librosa.stft(y))
      S_filter = librosa.decompose.nn_filter(S_full, aggregate=np.median, metric='cosine', width=int(librosa.time_to_frames(2, sr=sr)))
      S_filter = np.minimum(S_full, S_filter)

      # Create masks for vocals and background
      margin_i, margin_v = 3, 11
      power = 3
      mask_i = librosa.util.softmask(S_filter, margin_i * (S_full - S_filter), power=power)
      mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power)
      S_foreground = mask_v * S_full
      S_background = mask_i * S_full

      # Save background and vocals files
      background_path = os.path.join(cache_folder, f'{hash}-background.wav')
      vocals_path = os.path.join(cache_folder, f'{hash}-vocals.wav')

      y_foreground = librosa.istft(S_foreground * phase)
      y_background = librosa.istft(S_background * phase)
      
      foreground_save_path = os.path.join(cache_folder, f'{name}-foreground.wav')
      background_save_path = os.path.join(cache_folder, f'{name}-background.wav')

      sf.write(foreground_save_path, y_foreground, sr)
      sf.write(background_save_path, y_background, sr)

      return background_path, vocals_path

  @app.get("/api/split")
  def split_audio_get(name: str, url: str):
      try:
          file_path = download_file(url, MEDIA_CACHE_DIR)
          background, vocals = separate_vocals_and_save(name, file_path)
          return {"background": background, "vocals": vocals}
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))

  @app.post("/api/split")
  async def split_audio_post(name: str = Form(...), file: UploadFile = File(...)):
      try:
          contents = await file.read()
          with open('.cache/' + file.filename, 'wb') as f:
              f.write(contents)
          background, vocals = separate_vocals_and_save(name, f.name)
          return {"background": background, "vocals": vocals}
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))
