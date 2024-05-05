import EventObject from "../../../elements/src/EventObject";
import { getAudioContext } from "../../../elements/src/managers/AudioManager";

export class AudioClock extends EventObject<
  "start" | "stop" | "pause" | "render"
> {
  domElement: HTMLDivElement;
  bpmInput: HTMLInputElement;
  playPauseButton: HTMLButtonElement;
  stopButton: HTMLButtonElement;
  currentTimeDisplay: HTMLSpanElement;
  private _isPlaying: boolean = false;
  private _bpm: number = 100;
  private startTime: number | null = null;

  get currentBeat(): number {
    const elapsedTime =
      this.startTime !== null
        ? getAudioContext().currentTime - this.startTime
        : 0;
    return elapsedTime * (this._bpm / 60);
  }

  get isPlaying(): boolean {
    return this._isPlaying;
  }

  get bpm(): number {
    return this._bpm;
  }

  constructor() {
    super();

    this.domElement = document.createElement("div");
    this.domElement.classList.add("audio-clock");

    this.bpmInput = document.createElement("input");
    this.bpmInput.classList.add("audio-clock-bpm");
    this.bpmInput.type = "number";
    this.bpmInput.value = "120";

    this.bpmInput.addEventListener("input", () => {
      this._bpm = parseFloat(this.bpmInput.value);
    });
    this.domElement.appendChild(this.bpmInput);
    this.playPauseButton = document.createElement("button");
    this.playPauseButton.classList.add("audio-clock-play-pause");
    this.playPauseButton.textContent = "Play";

    this.playPauseButton.addEventListener("click", () => {
      if (this._isPlaying) {
        this.stop();
      } else {
        getAudioContext();
        this.play();
      }
    });
    this.domElement.appendChild(this.playPauseButton);
    this.stopButton = document.createElement("button");
    this.stopButton.classList.add("audio-clock-stop");
    this.stopButton.textContent = "Stop";

    this.stopButton.addEventListener("click", () => {
      this.stop();
    });
    this.domElement.appendChild(this.stopButton);
    this.currentTimeDisplay = document.createElement("span");
    this.currentTimeDisplay.classList.add("audio-clock-time");
    this.currentTimeDisplay.textContent = "01:01";

    this.domElement.appendChild(this.currentTimeDisplay);
  }

  private start(): void {
    this.fireEvent("start");
    this.startTime = getAudioContext().currentTime;
    console.log("Started at", this.startTime);
    requestAnimationFrame(this.render.bind(this));
  }

  private render(): void {
    this.updateCurrentTimeDisplay(this.currentBeat);
    this.fireEvent("render");
    if (this._isPlaying) requestAnimationFrame(this.render.bind(this));
  }

  private stop(): void {
    this.fireEvent("stop");
    this._isPlaying = false;
    this.startTime = null;
    this.updateCurrentTimeDisplay(this.currentBeat);
    this.playPauseButton.textContent = "Play";
  }

  private play(): void {
    if (this._isPlaying) {
      this.stop(); // TODO: This should be a pause
    } else {
      this.start();
    }
    this._isPlaying = !this._isPlaying;
    this.playPauseButton.textContent = this._isPlaying ? "Pause" : "Play";
  }

  private updateCurrentTimeDisplay(beat: number): void {
    const bars = Math.floor(beat / 4) + 1;
    const beats = Math.floor(beat % 4) + 1;
    const timeString = `${bars
      .toString()
      .padStart(2, "0")}:${beats.toString()}`;
    this.currentTimeDisplay.textContent = timeString;
  }
}
