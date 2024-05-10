import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { Composition } from "../Composition";
import { AudioClock } from "../components/AudioClock";
import { AudioCursor, ICursorTimeline } from "../components/AudioCursor";
import { GridItem } from "../components/Grid";
import { PatternTrack } from "../components/PatternTrack";

export class PatternWindow
  extends DraggableWindow<"select">
  implements ICursorTimeline
{
  readonly timeline: HTMLDivElement;
  readonly cursor: AudioCursor;
  readonly tracks: PatternTrack[] = [];
  beatWidth = 50;

  constructor(
    readonly audioClock: AudioClock,
    readonly composition: Composition
  ) {
    const container = document.createElement("div");
    container.classList.add("pattern-track-container");

    super("Pattern", true, container);
    this.timeline = container;
    this.setSize(800, 400);

    this.cursor = new AudioCursor(this);
    container.appendChild(this.cursor.domElement);
  }

  addTrack(name: string, events: GridItem[] = []) {
    const track = new PatternTrack(name, this.audioClock, events);

    if (this.tracks.length === 0) {
      track.selected = true;
    }

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

  trigger(note: number) {
    for (const track of this.tracks) {
      if (track.selected) track.trigger(note);
    }
  }

  play() {
    if (this.audioClock.startTime == null) {
      throw new Error("Audio clock start time is not set");
    }
    for (const track of this.tracks) {
      for (const event of track.events) {
        track.trigger(event.note, this.audioClock.startTime + event.start * 60 / this.audioClock.bpm);
      }
    }
  }
}
