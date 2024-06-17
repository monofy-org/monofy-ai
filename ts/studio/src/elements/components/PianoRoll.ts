import EventObject from "../../../../elements/src/EventObject";
import { GraphicsHelpers } from "../../abstracts/GraphicsHelpers";
import { AudioClock } from "./AudioClock";
import { AudioCursor, ICursorTimeline } from "./AudioCursor";
import { Grid, GridItem } from "./Grid";
import { IKeyboardEvent } from "./Keyboard";
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
      if (this.track) {
        GraphicsHelpers.renderSequence(
          this.track.preview.canvas,
          this.track.events,
          this.color,
          this.beatWidth
        );
      } else {
        console.warn("No track loaded!");
      }
    });
    this.grid.on("update", () => {
      if (this.track) {
        GraphicsHelpers.renderSequence(
          this.track.preview.canvas,
          this.track.events,
          this.color,
          this.beatWidth
        );
      }
    });
    this.grid.on("release", (item) => {      
      this.track!.release((item as GridItem).note);
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
