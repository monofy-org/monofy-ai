import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { Composition } from "../Composition";
import { AudioClock } from "../components/AudioClock";
import { AudioCursor } from "../components/AudioCursor";
import { GridItem } from "../components/Grid";
import { PatternTrack } from "../components/PatternTrack";
import { ICursorTimeline } from "../ICursorTimeline";

export class PatternWindow
  extends DraggableWindow<"select">
  implements ICursorTimeline
{
  readonly timeline: HTMLDivElement;
  readonly cursor: AudioCursor;
  readonly tracks: PatternTrack[] = [];

  constructor(
    readonly audioClock: AudioClock,
    readonly composition: Composition
  ) {
    const container = document.createElement("div");
    container.classList.add("pattern-track-container");

    super("Pattern", true, container);
    this.timeline = container;
    this.setSize(800, 400);

    this.cursor = new AudioCursor(audioClock, this);
  }

  addTrack(name: string, events: GridItem[] = []) {
    const track = new PatternTrack(name, events);

    track.on("select", (selectedTrack) => {
      this.emit("select", selectedTrack);
    });
    this.tracks.push(track);
    this.timeline.appendChild(track.domElement);

    return track;
  }

  removeTrack(track: PatternTrack) {
    const index = this.tracks.indexOf(track);
    if (index !== -1) {
      this.tracks.splice(index, 1);
      this.timeline.removeChild(track.domElement);
    }
  }
}
