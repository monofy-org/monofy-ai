import { triggerActive } from "../../../../elements/src/animation";
import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { SelectableElement } from "../../../../elements/src/elements/SelectableElement";
import type { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import type { Instrument } from "../../abstracts/Instrument";
import { InstrumentWindow } from "../../abstracts/InstrumentWindow";
import type { IEvent, ISequence } from "../../schema";
import type { ProjectUI } from "../ProjectUI";
import { MuteSoloButtons } from "./MuteSoloButtons";

export class PatternTrackInstrument extends SelectableElement {
  readonly indicator: HTMLDivElement;
  readonly muteSoloButtons: MuteSoloButtons;

  constructor(readonly track: PatternTrack) {
    console.assert(track, "PatternTrackInstrument track is null or undefined");

    super("div", "pattern-track-instrument");

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

    const win = this.track.window;

    edit.addEventListener("pointerdown", () => {
      if (win.isVisible) {
        win.close();
      } else {
        win.show();
      }
    });
    win.on("close", () => {
      edit.classList.toggle("active", false);
    });
    win.on("open", () => {
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
  private readonly _window: InstrumentWindow;

  get window() {
    return this._window;
  }

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
    readonly ui: ProjectUI,
    public instrument: Instrument,
    public events: IEvent[],
    readonly buttonGroup: SelectableGroup,
    readonly previewGroup: SelectableGroup
  ) {
    super("div", "pattern-track");

    this._window = new instrument.Window(ui, instrument);

    this.button = new PatternTrackInstrument(this);
    this.buttonGroup.addSelectable(this.button, false);
    this.preview = new PatternTrackPreview(this);

    console.log("New window", this.window);

    this.domElement.appendChild(this.button.domElement);
    this.domElement.appendChild(this.preview.domElement);
  }

  trigger(
    note: number,
    beat: number = this.ui.project.audioClock.currentBeat,
    velocity = 1.0
  ) {
    const source = this.instrument.trigger(note, beat, velocity);
    if (beat > 0) {
      this.ui.project.audioClock.scheduleEventAtBeat(() => {
        triggerActive(this.button.indicator);
      }, beat);
    } else {
      triggerActive(this.button.indicator);
    }
    return source;
  }

  release(note: number, beat: number = this.ui.project.audioClock.currentBeat) {
    this.instrument.release(note, beat);
  }

  load(sequence: ISequence) {
    this.events = sequence.events;
  }
}
