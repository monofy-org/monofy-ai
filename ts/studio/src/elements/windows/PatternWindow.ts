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
  readonly trackContainer: HTMLDivElement;
  readonly cursor: AudioCursor;
  readonly tracks: PatternTrack[] = [];
  readonly timeline: HTMLDivElement;
  beatWidth = 20;

  constructor(
    readonly audioClock: AudioClock,
    readonly composition: Composition
  ) {
    const container = document.createElement("div");
    container.classList.add("pattern-track-container");

    super("Pattern", true, container);
    this.trackContainer = container;
    this.setSize(800, 400);

    this.timeline = document.createElement("div");
    this.timeline.style.position = "absolute";
    this.timeline.style.width = "calc(100% - 88px)";
    this.timeline.style.height = "100%";
    this.timeline.style.left = "88px";
    this.timeline.style.pointerEvents = "none";
    this.timeline.style.overflow = "hidden";

    this.cursor = new AudioCursor(this);
    this.timeline.appendChild(this.cursor.domElement);

    const canvas = this.domElement.querySelector("canvas");
    if (canvas) this.beatWidth = canvas.width / 16;

    this.on("resize", () => {
      this.beatWidth = this.domElement.clientWidth / 16;
    });

    audioClock.on("start", () => {
      if (!audioClock.startTime) {
        throw new Error("Audio clock start time is not set");
      }
      for (const track of this.tracks) {
        track.playback();
      }
    });

    // container.appendChild(this.cursor.domElement);
  }

  addTrack(name: string, events: GridItem[] = []) {
    const track = new PatternTrack(name, this.audioClock, events);

    if (this.tracks.length === 0) {
      track.selected = true;
      track.domElement.appendChild(this.timeline);
    }

    track.on("select", (selectedTrack) => {
      this.emit("select", selectedTrack);
    });
    this.tracks.push(track);
    this.trackContainer.appendChild(track.domElement);

    return track;
  }

  removeTrack(track: PatternTrack) {
    const index = this.tracks.indexOf(track);
    if (index !== -1) {
      this.tracks.splice(index, 1);
      this.trackContainer.removeChild(track.domElement);
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
        track.trigger(
          event.note,
          this.audioClock.startTime + (event.start * 60) / this.audioClock.bpm
        );
      }
    }
  }
}
