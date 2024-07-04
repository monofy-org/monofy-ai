export abstract class AudioImporter {
  static decodeAudioData(
    arrayBuffer: ArrayBuffer,
    audioContext: AudioContext
  ): Promise<AudioBuffer> {
    return new Promise((resolve, reject) => {
      audioContext.decodeAudioData(
        arrayBuffer,
        (buffer) => resolve(buffer),
        (error) => reject(error)
      );
    });
  }

  static async loadUrl(url: string, audioContext: AudioContext) {
    try {
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();
      return await AudioImporter.decodeAudioData(arrayBuffer, audioContext);
    } catch (error) {
      console.error("Error loading audio file:", error);
    }
  }

  static async loadFile(file: File, audioContext: AudioContext) {
    try {
      const arrayBuffer = await file.arrayBuffer();
      return await AudioImporter.decodeAudioData(arrayBuffer, audioContext);
    } catch (error) {
      console.error("Error loading audio file:", error);
    }
  }

  static loadBlob(
    blob: Blob,
    audioContext: AudioContext
  ): Promise<AudioBuffer> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const arrayBuffer = reader.result as ArrayBuffer;
        AudioImporter.decodeAudioData(arrayBuffer, audioContext)
          .then((audioBuffer) => {
            resolve(audioBuffer);
          })
          .catch((error) => {
            reject(error);
          });
      };
      reader.onerror = (error) => {
        reject(error);
      };
      reader.readAsArrayBuffer(blob);
    });
  }
}
