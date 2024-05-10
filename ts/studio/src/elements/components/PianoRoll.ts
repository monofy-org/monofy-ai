import EventObject from "../../../../elements/src/EventObject";
import { getAudioContext } from "../../../../elements/src/managers/AudioManager";
import { ICursorTimeline } from "../ICursorTimeline";
import { AudioClock } from "./AudioClock";
import { AudioCursor } from "./AudioCursor";
import { Grid } from "./Grid";
import { PatternTrack } from "./PatternTrack";
import { SideKeyboard } from "./SideKeyboard";

export class PianoRoll
  extends EventObject<"update">
  implements ICursorTimeline
{
  readonly domElement: HTMLDivElement;
  readonly grid: Grid;
  readonly sideKeyboard: SideKeyboard;
  readonly cursor: AudioCursor;
  readonly timeline: HTMLElement;
  cursorUpdateInterval: number | object | null = null;
  scheduledSources: AudioBufferSourceNode[] = [];
  track: PatternTrack | null = null;
  color = "#7979ce";
  beatWidth = 100;

  constructor(readonly audioClock: AudioClock) {
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
      this.emit("update", this);
    });

    this.timeline = this.grid.domElement;

    this.cursor = new AudioCursor(this);
    this.cursor.domElement.classList.add("audio-cursor");
    this.grid.domElement.appendChild(this.cursor.domElement);
  }

  scheduleAudioEvents() {
    const ctx = getAudioContext();
    this.track?.events.forEach((note) => {
      const bufferSource = ctx.createBufferSource();
      bufferSource.buffer = note.audio;
      bufferSource.connect(ctx.destination);
      bufferSource.start(
        ctx.currentTime + (note.start * 60) / this.audioClock.bpm
      );
      this.scheduledSources.push(bufferSource);
    });
  }

  load(track: PatternTrack) {
    this.track = track;
    this.grid.load(track);
  }
}
