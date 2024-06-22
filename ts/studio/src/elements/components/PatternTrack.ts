import { triggerActive } from "../../../../elements/src/animation";
import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { SelectableElement } from "../../../../elements/src/elements/SelectableElement";
import type { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import type { Instrument } from "../../abstracts/Instrument";
import type { IEvent, ISequence } from "../../schema";
import type { Project } from "../Project";
import { MuteSoloButtons } from "./MuteSoloButtons";
import type { ISourceEvent } from "./SamplerSlot";

export class PatternTrackInstrument extends SelectableElement {
  readonly indicator: HTMLDivElement;
  readonly muteSoloButtons: MuteSoloButtons;

  constructor(
    readonly track: PatternTrack,
    buttonGroup: SelectableGroup
  ) {
    console.assert(track, "PatternTrackInstrument track is null or undefined");

    super(buttonGroup, "div", "pattern-track-instrument");

    this.domElement.classList.add("pattern-track-panel");

    const labelAndButtons = document.createElement("div");
    labelAndButtons.classList.add("pattern-track-label-and-buttons");

    const textLabel = document.createElement("div");
    textLabel.textContent = track.name;
    labelAndButtons.appendChild(textLabel);

    this.muteSoloButtons = new MuteSoloButtons();
    this.muteSoloButtons.on("change", () => {
      track.mute = this.muteSoloButtons.mute;
      track.solo = this.muteSoloButtons.solo;
    });

    labelAndButtons.appendChild(this.muteSoloButtons.domElement);

    this.domElement.appendChild(labelAndButtons);

    const edit = document.createElement("div");
    edit.classList.add("track-button");
    edit.classList.add("track-edit-button");
    edit.textContent = "e";
    edit.addEventListener("pointerdown", () => {
      if (this.track.instrument.window.isVisible) {
        this.track.instrument.window.close();
      } else {
        track.instrument.window.show();
      }
    });
    this.track.instrument.window.on("close", () => {
      edit.classList.toggle("active", false);
    });
    this.track.instrument.window.on("open", () => {
      edit.classList.toggle("active", true);
    });
    this.muteSoloButtons.domElement.appendChild(edit);

    this.indicator = document.createElement("div");
    this.indicator.classList.add("pattern-track-indicator");
    this.domElement.appendChild(this.indicator);
  }
}

export class PatternTrackPreview extends BaseElement {
  private _canvas: HTMLCanvasElement;

  get canvas() {
    return this._canvas;
  }

  constructor(readonly track: PatternTrack) {
    console.assert(track, "PatternTrackPreview track is null or undefined");

    super("div", "pattern-track-preview");
    this._canvas = document.createElement("canvas");
    this._canvas.height = 60;
    this._canvas.width = 1280;
    this._canvas.classList.add("pattern-track-pattern");

    this.domElement.append(this._canvas);
  }
}

export class PatternTrack extends BaseElement implements ISequence {
  readonly button: PatternTrackInstrument;
  readonly preview: PatternTrackPreview;
  port: number | null = null;
  channel: number = 0;

  get mute() {
    return this.button.muteSoloButtons.mute;
  }

  set mute(value: boolean) {
    this.button.muteSoloButtons.mute = value;
  }

  get solo() {
    return this.button.muteSoloButtons.solo;
  }

  set solo(value: boolean) {
    this.button.muteSoloButtons.solo = value;
  }

  get name() {
    return this.instrument.name;
  }

  constructor(
    readonly project: Project,
    public instrument: Instrument,
    public events: IEvent[],
    readonly buttonGroup: SelectableGroup,
    readonly previewGroup: SelectableGroup
  ) {
    super("div", "pattern-track");

    this.button = new PatternTrackInstrument(this, this.buttonGroup);
    this.preview = new PatternTrackPreview(this);

    this.domElement.appendChild(this.button.domElement);
    this.domElement.appendChild(this.preview.domElement);
  }

  trigger(note: number, beat: number = this.project.audioClock.currentBeat) {
    const source = this.instrument.trigger(note, this.channel, beat);
    if (beat > 0) {
      this.project.audioClock.scheduleEventAtBeat(() => {
        triggerActive(this.button.indicator);
      }, beat);
    } else {
      triggerActive(this.button.indicator);
    }
    return source;
  }

  release(note: number, beat: number = this.project.audioClock.currentBeat) {
    this.instrument.release(note, beat);
  }

  load(sequence: ISequence) {
    this.events = sequence.events;
  }

  playback() {
    if (!this.project.audioClock.isPlaying) {
      console.warn("Playback cancelled");
      return;
    }

    let sourceEvent: ISourceEvent | undefined = undefined;

    for (const event of this.events) {
      if (!(event.note || event.note === 0)) {
        console.warn("Missing note property in event object");
        continue;
      }

      sourceEvent = this.trigger(event.note, event.start);
      if (event.domElement) {
        this.project.audioClock.scheduleEventAtBeat(
          () => event.domElement!.classList.toggle("active", true),
          event.start
        );
        this.project.audioClock.scheduleEventAtBeat(
          () => event.domElement!.classList.toggle("active", false),
          event.start + event.duration
        );
      }
    }

    if (typeof sourceEvent !== "undefined" && "source" in sourceEvent) {
      sourceEvent.source.onended = () => {
        const nextLoop =
          this.project.audioClock.currentBeat +
          (4 - (this.project.audioClock.currentBeat % 4));
        this.project.audioClock.scheduleEventAtBeat(() => {
          this.project.audioClock.restart();
          this.playback();
        }, nextLoop);
      };
    }
  }
}
