import { getAudioContext } from "../../../elements/src/managers/AudioManager";

export class AudioCanvas {
  domElement: HTMLDivElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  buffer: AudioBuffer | null = null;

  constructor() {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("audio-canvas");

    this.canvas = document.createElement("canvas");
    this.canvas.width = 800;
    this.canvas.height = 200;
    this.domElement.appendChild(this.canvas);

    this.canvas.addEventListener("click", () => {
      if (this.buffer) {
        this.playBuffer(this.buffer);
      }
    });

    this.ctx = this.canvas.getContext("2d") as CanvasRenderingContext2D;
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

  playBuffer(audioBuffer: AudioBuffer) {
    const ctx = getAudioContext();
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(ctx.destination);
    source.start();
  }

  loadBuffer(audioBuffer: AudioBuffer) {
    this.buffer = audioBuffer;

    const channelData = audioBuffer.getChannelData(0);
    const bufferLength = channelData.length;
    const sliceWidth = this.canvas.width / bufferLength;

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.beginPath();
    this.ctx.moveTo(0, this.canvas.height / 2);

    for (let i = 0; i < bufferLength; i++) {
      const x = i * sliceWidth;
      const y = ((channelData[i] + 1) * this.canvas.height) / 2;

      this.ctx.lineTo(x, y);
    }

    this.ctx.lineTo(this.canvas.width, this.canvas.height / 2);
    this.ctx.stroke();
  }
}
