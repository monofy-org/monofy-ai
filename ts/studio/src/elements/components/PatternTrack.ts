import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { triggerActive } from "../../../../elements/src/animation";
import { IEventItem, ISequence } from "../../schema";
import { ISourceEvent } from "./SamplerSlot";
import { Project } from "../Project";
import { Instrument } from "../../abstracts/Instrument";

export class PatternTrack
  extends BaseElement<"update" | "select" | "edit">
  implements ISequence
{
  private _canvas: HTMLCanvasElement;
  private readonly _instrumentPanel: HTMLDivElement;
  private readonly _indicator: HTMLDivElement;
  port: number | null = null;
  channel: number = 0;
  events: IEventItem[] = [];

  get canvas() {
    return this._canvas;
  }

  get name() {
    return this.instrument.name;
  }

  private _selected: boolean = false;
  get selected() {
    return this._selected;
  }
  set selected(value: boolean) {
    console.log("PatternTrack set selected", value);
    this._selected = value;
    this._instrumentPanel.classList.toggle("selected", value);
  }

  constructor(
    readonly project: Project,
    public instrument: Instrument
  ) {
    super("div", "pattern-track");

    this._instrumentPanel = document.createElement("div");
    this._instrumentPanel.classList.add("pattern-track-panel");
    this._instrumentPanel.addEventListener("pointerdown", () => {
      this.instrument.window.show();
    });

    const label = document.createElement("div");
    this._instrumentPanel.appendChild(label);
    label.textContent = instrument.name;

    const buttons = document.createElement("div");
    buttons.classList.add("pattern-track-buttons");
    this._instrumentPanel.appendChild(buttons);

    const edit = document.createElement("button");
    edit.textContent = "e";
    buttons.appendChild(edit);

    const mute = document.createElement("button");
    mute.textContent = "M";
    buttons.appendChild(mute);

    const solo = document.createElement("button");
    solo.textContent = "S";
    buttons.appendChild(solo);

    this._indicator = document.createElement("div");
    this._indicator.classList.add("pattern-track-indicator");
    buttons.appendChild(this._indicator);

    buttons.appendChild(mute);
    buttons.appendChild(solo);

    this._canvas = document.createElement("canvas");
    this._canvas.height = 100;
    this._canvas.width = 1600;
    this._canvas.classList.add("pattern-track-pattern");

    this._canvas.addEventListener("pointerdown", () => {
      this.emit("edit", this);
    });

    this._instrumentPanel.addEventListener("pointerdown", () => {
      this.emit("select", this);
    });

    this.domElement.appendChild(this._instrumentPanel);
    this.domElement.appendChild(this._indicator);
    this.domElement.appendChild(this._canvas);
  }

  trigger(note: number, beat = 0) {
    const source = this.instrument.trigger(note, this.channel, beat);
    if (beat > 0) {
      this.project.audioClock.scheduleEventAtBeat(() => {
        triggerActive(this._indicator);
      }, beat);
    } else {
      triggerActive(this._indicator);
    }
    return source;
  }

  release(note: number, beat = 0) {
    console.warn("TODO: PatternTrack release", note, beat);
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
