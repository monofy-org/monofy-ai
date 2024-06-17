import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import { Instrument } from "../../abstracts/Instrument";
import { IPattern } from "../../schema";
import { Project } from "../Project";
import { ProjectUI } from "../ProjectUI";
import { AudioClock } from "../components/AudioClock";
import { AudioCursor, ICursorTimeline } from "../components/AudioCursor";
import {
  PatternTrack,
  PatternTrackInstrument,
  PatternTrackPreview,
} from "../components/PatternTrack";

export class PatternWindow
  extends DraggableWindow<"select" | "edit">
  implements ICursorTimeline
{
  readonly trackContainer: HTMLDivElement;
  readonly cursor: AudioCursor;
  readonly timeline: HTMLDivElement;
  readonly tracks: PatternTrack[] = [];
  readonly patternPreviews: SelectableGroup<PatternTrackPreview>;
  readonly buttons: SelectableGroup<PatternTrackInstrument>;

  get audioClock(): AudioClock {
    return this.ui.project.audioClock;
  }

  beatWidth = 20;

  constructor(public ui: ProjectUI) {
    const container = document.createElement("div");
    container.classList.add("pattern-track-container");

    super({
      title: "Pattern",
      persistent: true,
      content: container,
      width: 400,
      height: 400,
    });
    this.trackContainer = container;

    this.timeline = this.trackContainer;

    this.cursor = new AudioCursor(this);
    this.timeline.appendChild(this.cursor.domElement);

    const canvas = this.domElement.querySelector("canvas");
    if (canvas) this.beatWidth = canvas.width / 16;

    this.on("resize", () => {
      this.beatWidth = this.domElement.clientWidth / 16;
    });

    ui.project.audioClock.on("start", () => {
      if (!ui.project.audioClock.startTime) {
        throw new Error("Audio clock start time is not set");
      }
      for (const track of this.tracks) {
        track.playback();
      }
    });

    this.addPattern("Pattern 1");

    this.patternPreviews = new SelectableGroup<PatternTrackPreview>();
    this.buttons = new SelectableGroup<PatternTrackInstrument>();
  }

  loadProject(project: Project) {
    for (const track of this.tracks) {
      this.trackContainer.removeChild(track.domElement);
    }
    this.tracks.length = 0;
    for (let i = 0; i < project.instruments.length; i++) {
      const track = this.addTrack(project.instruments[i]);
      track.load(project.patterns[0].sequences[i]);
      if (i === 0) {
        track.button.selected = true;
      }
    }
  }

  addPattern(name: string) {
    console.log("Created pattern:", name);
    const pattern: IPattern = { name, sequences: [] };
    this.ui.project.patterns.push(pattern);
  }

  addTrack(instrument: Instrument) {
    console.log("Add track + instrument:", instrument);

    const track = new PatternTrack(
      this.ui.project,
      instrument,
      this.buttons,
      this.patternPreviews
    );

    console.assert(
      track.button instanceof PatternTrackInstrument,
      "button is not an instance of PatternTrackInstrument"
    );
    console.assert(
      track.preview instanceof PatternTrackPreview,
      "preview is not an instance of PatternTrackPreview"
    );

    track.preview.domElement.addEventListener("pointerdown", () => {
      this.ui.pianoRollWindow.loadTrack(track);
      this.ui.pianoRollWindow.show();
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
      this.patternPreviews.removeSelectable(track.preview);
      this.buttons.removeSelectable(track.button);
    }
  }

  trigger(note: number, beat: number = this.audioClock.currentBeat) {
    for (const track of this.tracks) {
      if (track.button.selected) {
        track.trigger(note, beat);
      }
    }
  }

  release(note: number, beat: number = this.audioClock.currentBeat) {
    for (const track of this.tracks) {
      if (track.button.selected) {
        track.release(note, beat);
      }
    }
  }

  play() {
    if (this.audioClock.startTime == null) {
      throw new Error("Audio clock start time is not set");
    }
    for (const track of this.tracks) {
      for (const event of track.events) {
        track.trigger(
          event.note!
          ,
          this.audioClock.startTime + (event.start * 60) / this.audioClock.bpm
        );
      }
    }
  }
}
