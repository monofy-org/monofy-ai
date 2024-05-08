import EventObject from "../../../../elements/src/EventObject";
import { getAudioContext } from "../../../../elements/src/managers/AudioManager";
import { AudioClock } from "./AudioClock";
import { Grid } from "./Grid";
import { PatternTrack } from "./PatternTrack";
import { SideKeyboard } from "./SideKeyboard";

export class PianoRoll extends EventObject<"update"> {
  domElement: HTMLDivElement;
  grid: Grid;
  sideKeyboard: SideKeyboard;
  cursor: HTMLDivElement;
  cursorUpdateInterval: number | object | null = null;
  scheduledSources: AudioBufferSourceNode[] = [];
  track: PatternTrack | null = null;
  color = "#7979ce";

  constructor(private readonly clock: AudioClock) {
    super();

    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll");

    this.sideKeyboard = new SideKeyboard();
    this.domElement.appendChild(this.sideKeyboard.domElement);

    this.grid = new Grid();
    this.domElement.appendChild(this.grid.domElement);
    this.grid.on("update", () => {
      if (this.track) {
        this.grid.renderToCanvas(this.track.canvas, this.color);
      }
      this.fireEvent("update", this);
    });

    this.cursor = document.createElement("div");
    this.cursor.classList.add("piano-roll-cursor");
    this.grid.domElement.appendChild(this.cursor);

    clock.on("start", () => {
      if (this.cursorUpdateInterval) {
        clearInterval(this.cursorUpdateInterval as number);
      }
      this.cursor.style.transform = "translateX(0)";
      this.cursor.style.display = "block";
      this.cursor.parentElement?.appendChild(this.cursor);
      this.scheduleAudioEvents();
    });

    clock.on("update", () => {
      this.updateCursor();
    });

    clock.on("stop", () => {
      this.cursor.style.display = "none";
      this.scheduledSources.forEach((source) => source.stop());
      this.scheduledSources = [];
    });
  }

  updateCursor() {
    if (this.clock.isPlaying) {
      this.cursor.style.transform = `translateX(${
        this.clock.currentBeat * this.grid.beatWidth
      }px)`;
    }
  }

  scheduleAudioEvents() {
    const ctx = getAudioContext();
    this.track?.events.forEach((note) => {
      const bufferSource = ctx.createBufferSource();
      bufferSource.buffer = note.audio;
      bufferSource.connect(ctx.destination);
      bufferSource.start(ctx.currentTime + (note.start * 60) / this.clock.bpm);
      this.scheduledSources.push(bufferSource);
    });
  }

  load(track: PatternTrack) {
    this.track = track;
    this.grid.load(track);
  }
}
