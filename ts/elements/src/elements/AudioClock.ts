import EventObject from "../EventObject";

export class AudioClock extends EventObject<"start" | "stop" | "pause"> {
  domElement: HTMLDivElement;
  bpmInput: HTMLInputElement;
  playPauseButton: HTMLButtonElement;
  stopButton: HTMLButtonElement;
  currentTimeDisplay: HTMLSpanElement;  
  private _isPlaying: boolean = false;
  private bpm: number = 100;
  private intervalId: any = null;
  private startTime: number | null = null;

  get currentBeat(): number {
    const elapsedTime = this.startTime
      ? (Date.now() - this.startTime) / 1000
      : 0;
    return elapsedTime * (this.bpm / 60);
  }

  get isPlaying(): boolean {
    return this._isPlaying;
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
      this.bpm = parseFloat(this.bpmInput.value);
    });
    this.domElement.appendChild(this.bpmInput);
    this.playPauseButton = document.createElement("button");
    this.playPauseButton.classList.add("audio-clock-play-pause");
    this.playPauseButton.textContent = "Play";

    this.playPauseButton.addEventListener("click", () => {
      if (this._isPlaying) {
        this.stop();
      } else {
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
    this.startTime = Date.now();
    this.intervalId = setInterval(
      () => {        
        this.updateCurrentTimeDisplay(this.currentBeat);
      },
      250 // Update every 250ms
    );
  }

  private stop(): void {
    this.fireEvent("stop");
    clearInterval(this.intervalId);
    this._isPlaying = false;
    this.intervalId = null;
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
    const timeString = `${bars.toString().padStart(2, "0")}:${beats
      .toString()
      .padStart(2, "0")}`;
    this.currentTimeDisplay.textContent = timeString;
  }
}
