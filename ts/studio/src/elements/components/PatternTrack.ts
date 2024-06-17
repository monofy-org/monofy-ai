import { triggerActive } from "../../../../elements/src/animation";
import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { SelectableElement } from "../../../../elements/src/elements/SelectableElement";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import { Instrument } from "../../abstracts/Instrument";
import { IEvent, ISequence } from "../../schema";
import { Project } from "../Project";
import { ISourceEvent } from "./SamplerSlot";

export class PatternTrackInstrument extends SelectableElement {
  readonly indicator: HTMLDivElement;

  constructor(
    readonly track: PatternTrack,
    buttonGroup: SelectableGroup<PatternTrackInstrument>
  ) {
    console.assert(track, "PatternTrackInstrument track is null or undefined");

    super(buttonGroup, "div", "pattern-track-instrument");

    document.createElement("div");
    this.domElement.classList.add("pattern-track-panel");

    const textLabel = document.createElement("div");
    textLabel.textContent = track.name;

    const buttons = document.createElement("div");
    buttons.classList.add("pattern-track-buttons");

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
    buttons.appendChild(edit);

    const mute = document.createElement("div");
    mute.classList.add("track-button");
    mute.classList.add("track-mute-button");
    mute.textContent = "M";
    mute.addEventListener("pointerdown", () => {
      mute.classList.toggle("active");
    });
    buttons.appendChild(mute);

    const solo = document.createElement("div");
    solo.classList.add("track-button");
    solo.classList.add("track-solo-button");
    solo.textContent = "S";
    solo.addEventListener("pointerdown", () => {
      solo.classList.toggle("active");
    });
    buttons.appendChild(solo);

    buttons.appendChild(mute);
    buttons.appendChild(solo);

    this.indicator = document.createElement("div");
    this.indicator.classList.add("pattern-track-indicator");
    buttons.appendChild(this.indicator);

    this.domElement.appendChild(textLabel);
    this.domElement.appendChild(buttons);
  }
}

export class PatternTrackPreview extends SelectableElement {
  private _canvas: HTMLCanvasElement;

  get canvas() {
    return this._canvas;
  }

  constructor(
    readonly track: PatternTrack,
    previewGroup: SelectableGroup<PatternTrackPreview>
  ) {
    console.assert(track, "PatternTrackPreview track is null or undefined");

    super(previewGroup, "div", "pattern-track-preview");
    this._canvas = document.createElement("canvas");
    this._canvas.height = 100;
    this._canvas.width = 1600;
    this._canvas.classList.add("pattern-track-pattern");

    this.domElement.append(this._canvas);
  }
}

export class PatternTrack extends BaseElement<"update"> implements ISequence {
  readonly button: PatternTrackInstrument;
  readonly preview: PatternTrackPreview;
  port: number | null = null;
  channel: number = 0;
  events: IEvent[] = [];

  get name() {
    return this.instrument.name;
  }

  constructor(
    readonly project: Project,
    public instrument: Instrument,
    readonly buttonGroup: SelectableGroup<PatternTrackInstrument>,
    readonly previewGroup: SelectableGroup<PatternTrackPreview>
  ) {
    super("div", "pattern-track");

    this.button = new PatternTrackInstrument(this, this.buttonGroup);
    this.preview = new PatternTrackPreview(this, this.previewGroup);

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
