import EventObject from "../../../../elements/src/EventObject";
import { AudioClock } from "./AudioClock";
import { AudioCursor, ICursorTimeline } from "./AudioCursor";
import type { GridItem } from "./Grid";
import { Grid } from "./Grid";
import type { IKeyboardEvent } from "./Keyboard";
import type { PatternTrack } from "./PatternTrack";
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
  track: PatternTrack | null = null;
  color = "#7979ce";
  beatWidth = 100;

  constructor(readonly audioClock: AudioClock) {
    super();

    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll");

    this.sideKeyboard = new SideKeyboard();
    this.domElement.appendChild(this.sideKeyboard.domElement);
    this.sideKeyboard.on("update", (event) => {
      const e = event as IKeyboardEvent;
      if (e.type == "press") this.track!.trigger(e.note);
      else if (e.type == "release") this.track!.release(e.note);
    });

    this.grid = new Grid();
    this.domElement.appendChild(this.grid.domElement);
    this.grid.on("select", (item) => {
      const note = item as GridItem;
      this.track!.trigger(note.note);
      // this.track?.preview.update();
    });
    this.grid.on("update", () => {
      this.track?.preview.update();
      this.emit("update");
    });
    this.grid.on("release", (item) => {
      if (item) this.track!.release((item as GridItem).note);
    });
    this.grid.linkElement(this.sideKeyboard.domElement);

    this.timeline = this.grid.domElement;

    this.cursor = new AudioCursor(this);
    this.cursor.domElement.classList.add("audio-cursor");
    this.grid.domElement.appendChild(this.cursor.domElement);
    this.cursor.domElement.style.marginLeft =
      this.sideKeyboard.domElement.offsetWidth + "px";

    this.grid.scrollTop = 1000;
  }

  load(track: PatternTrack) {
    if (!track) {
      console.warn("No track provided!");
      return;
    }
    this.track = track;
    this.grid.load(track);
  }
}
