import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { ISequence } from "../../schema";
import { SamplerWindow } from "../windows/SamplerWindow";
import { AudioClock } from "./AudioClock";
import { GridItem } from "./Grid";
import { SharedComponents } from "./SharedComponents";

export class PatternTrack
  extends BaseElement<"update" | "select">
  implements ISequence
{
  private _canvas: HTMLCanvasElement;
  private _name: string = "";
  private readonly _instrumentPanel: HTMLDivElement;
  private readonly _indicator: HTMLDivElement;
  output: "sampler" | "synth" | "midi" = "sampler";
  port: number | null = null;
  channel: number | null = null;

  get canvas() {
    return this._canvas;
  }

  get name() {
    return this._name;
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
    name: string,
    readonly audioClock: AudioClock,
    readonly events: GridItem[]
  ) {
    super("div", "pattern-track");

    this._instrumentPanel = document.createElement("div");
    this._instrumentPanel.classList.add("pattern-track-panel");

    const label = document.createElement("div");
    this._instrumentPanel.appendChild(label);
    label.textContent = name;

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
      this.emit("select", this);
    });

    this.domElement.appendChild(this._instrumentPanel);
    this.domElement.appendChild(this._indicator);
    this.domElement.appendChild(this._canvas);
  }

  async startAnimation() {
    this._indicator.classList.toggle("active", true);
    setTimeout(() => {
      this._indicator.classList.toggle("active", false);
    }, 100);
  }

  trigger(note: number, time = 0) {
    if (this.output === "sampler") {
      const sampler: SamplerWindow = SharedComponents.getComponent("sampler");
      sampler.trigger(note, this.channel, time);
      if (time > 0) {
        this.audioClock.scheduleEventAtTime(() => {
          this.startAnimation();
        }, time);
      } else {
        this.startAnimation();
      }
    }
  }
}
