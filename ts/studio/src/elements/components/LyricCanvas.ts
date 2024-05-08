import { getAudioContext } from "../../../../elements/src/managers/AudioManager";
import { AudioCanvas } from "../../../../elements/src/elements/AudioCanvas";

export class LyricCanvas extends AudioCanvas {
  constructor() {
    super();
  }

  async generateAudio(
    note: { label: string; note: number },
    preview: boolean = false
  ): Promise<AudioBuffer> {
    const buffer: AudioBuffer = await new Promise((resolve, reject) => {
      const req = {
        text: note.label,
        pitch: note.note,
        rate: 0.9,
      };
      fetch("/api/tts/edge", {
        method: "POST",
        body: JSON.stringify(req),
        headers: { "Content-Type": "application/json" },
      })
        .then((res) => res.arrayBuffer())
        .then((data) => {
          console.log("Audio buffer downloaded", data);
          getAudioContext().decodeAudioData(
            data,
            (audioBuffer: AudioBuffer) => {
              this.buffer = audioBuffer;
              if (preview) {
                this.playBuffer(audioBuffer);
              }

              resolve(audioBuffer);
            }
          );
        })
        .catch((error) => {
          console.error("Error downloading audio buffer", error);
          reject(error);
        });
    });

    this.loadBuffer(buffer);

    return buffer;
  }
}
