import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import { Instrument } from "../../abstracts/Instrument";
import { IEvent, IPattern } from "../../schema";
import { IProjectUpdateEvent, Project } from "../Project";
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
  readonly patternPreviews: SelectableGroup;
  readonly buttons: SelectableGroup;

  get audioClock(): AudioClock {
    return this.ui.project.audioClock;
  }

  beatWidth = 20;

  constructor(public ui: ProjectUI) {
    const container = document.createElement("div");
    container.classList.add("pattern-track-container");

    super(ui.container, {
      title: "Pattern",
      persistent: true,
      content: container,
      width: 400,
      height: 400,
    });
    this.trackContainer = container;

    this.timeline = this.trackContainer;

    this.cursor = new AudioCursor(this);
    this.cursor.domElement.style.marginLeft = "150px";
    this.timeline.appendChild(this.cursor.domElement);

    const canvas = this.domElement.querySelector("canvas");
    if (canvas) this.beatWidth = canvas.width / 16;

    this.on("resize", () => {
      this.beatWidth = this.domElement.clientWidth / 16;
    });

    ui.project.on("update", (e) => {
      const update = e as IProjectUpdateEvent;

      if (update.type === "project") {
        const project = update.value as Project;

        for (const track of this.tracks) {
          this.trackContainer.removeChild(track.domElement);
        }

        this.tracks.length = 0;

        if (this.ui.project.patterns.length === 0) {
          this.ui.project.patterns.push({
            name: "Pattern 1",
            tracks: [],
          });
        }

        for (let i = 0; i < project.instruments.length; i++) {
          if (!project.patterns[0].tracks[i]) {
            project.patterns[0].tracks[i] = { events: [] };
          }
          const track = this.addTrack(
            project.instruments[i],
            project.patterns[0].tracks[i].events || []
          );
          if (!project.patterns[0].tracks[i])
            track.load(project.patterns[0].tracks[i]);
          if (i === 0) {
            track.button.selected = true;
          }
        }
      }
    });

    this.patternPreviews = new SelectableGroup();
    this.buttons = new SelectableGroup<PatternTrackInstrument>();
  }

  loadPattern(pattern: IPattern) {
    this.setTitle(pattern.name);
    for (let i = 0; i < this.ui.project.instruments.length; i++) {
      pattern.tracks[i] = pattern.tracks[i] || [];
      this.tracks[i].load(pattern.tracks[i]);
    }
    if (!this.isVisible) this.show();
  }

  addTrack(instrument: Instrument, events: IEvent[]) {
    console.log("Add track + instrument:", instrument);

    const track = new PatternTrack(
      this.ui,
      instrument,
      events,
      this.buttons,
      this.patternPreviews
    );

    instrument.output.connect(this.ui.mixerWindow.mixer.channels[0].gainNode);

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
      this.buttons.removeSelectable(track.button);
    }
  }

  trigger(
    note: number,
    beat: number = this.audioClock.currentBeat,
    velocity = 1.0
  ) {
    for (const track of this.tracks) {
      if (track.button.selected) {
        track.trigger(note, beat, velocity);
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
}
