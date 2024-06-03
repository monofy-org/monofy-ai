import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import { Instrument } from "../../abstracts/Instrument";
import { IPattern } from "../../schema";
import { Project } from "../Project";
import { AudioClock } from "../components/AudioClock";
import { AudioCursor, ICursorTimeline } from "../components/AudioCursor";
import { PatternTrack } from "../components/PatternTrack";

export class PatternWindow
  extends DraggableWindow<"select" | "edit">
  implements ICursorTimeline
{
  readonly trackContainer: SelectableGroup<PatternTrack>;
  readonly cursor: AudioCursor;
  readonly timeline: HTMLDivElement;

  get tracks() {
    return this.trackContainer.items;
  }

  get audioClock(): AudioClock {
    return this.project.audioClock;
  }

  beatWidth = 20;
  patterns: IPattern[] = [
    {
      name: "Pattern 1",
      sequences: [],
    },
  ];

  constructor(readonly project: Project) {
    const tracks: PatternTrack[] = [];

    const container = new SelectableGroup<PatternTrack>(tracks);
    container.domElement.classList.add("pattern-track-container");

    super({
      title: "Pattern",
      persistent: true,
      content: container.domElement,
      width: 400,
      height: 400,
    });
    this.trackContainer = container;
    this.setSize(800, 400);

    project.patterns = this.patterns;

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

    project.audioClock.on("start", () => {
      if (!project.audioClock.startTime) {
        throw new Error("Audio clock start time is not set");
      }
      for (const track of this.trackContainer.items) {
        track.playback();
      }
    });

    this.addPattern("Pattern 1");

    // container.appendChild(this.cursor.domElement);
  }

  loadProject(project: Project) {
    this.patterns = project.patterns;
    this.trackContainer.items.forEach((track) => {
      this.trackContainer.remove(track);
    });
    this.trackContainer.items.length = 0;
    for (let i = 0; i < project.instruments.length; i++) {
      const track = this.addTrack(project.instruments[i]);
      track.load(project.patterns[0].sequences[i]);
    }
  }

  addPattern(name: string) {
    console.log("Created pattern:", name);
    const pattern: IPattern = { name, sequences: [] };
    this.patterns.push(pattern);
  }

  addTrack(instrument: Instrument) {
    console.log("Added track + instrument:", instrument);

    const track = new PatternTrack(this.project, instrument);
    if (this.trackContainer.items.length === 0) {
      // HACK: add cursor to first track      
      track.domElement.appendChild(this.timeline);
    }

    track.on("select", (selectedTrack) => {
      this.emit("select", selectedTrack);
    });

    track.on("edit", (selectedTrack) => {
      console.log("PatternWindow edit", selectedTrack);
      this.emit("edit", selectedTrack);
    });

    this.trackContainer.add(track);

    return track;
  }

  removeTrack(track: PatternTrack) {
    const index = this.trackContainer.items.indexOf(track);
    if (index !== -1) {
      this.trackContainer.items.splice(index, 1);
      this.trackContainer.remove(track);
    }
  }

  trigger(note: number) {
    for (const track of this.trackContainer.items) {
      if (track.selected) track.trigger(note);
    }
  }

  play() {
    if (this.audioClock.startTime == null) {
      throw new Error("Audio clock start time is not set");
    }
    for (const track of this.trackContainer.items) {
      for (const event of track.events) {
        track.trigger(
          event.note,
          this.audioClock.startTime + (event.start * 60) / this.audioClock.bpm
        );
      }
    }
  }
}
