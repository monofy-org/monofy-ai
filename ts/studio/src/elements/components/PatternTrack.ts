import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { ISequence } from "../../schema";
import { GridItem } from "./Grid";

export class PatternTrack
  extends BaseElement<"update" | "select">
  implements ISequence
{
  private _canvas: HTMLCanvasElement;
  private _name: string = "";

  get canvas() {
    return this._canvas;
  }

  get name() {
    return this._name;
  }

  constructor(
    name: string,
    readonly events: GridItem[]
  ) {
    super("div", "pattern-track");

    const instrumentPanel = document.createElement("div");
    instrumentPanel.classList.add("pattern-track-panel");

    const label = document.createElement("div");
    instrumentPanel.appendChild(label);
    label.textContent = name;

    const buttons = document.createElement("div");
    buttons.classList.add("pattern-track-buttons");
    instrumentPanel.appendChild(buttons);

    const edit = document.createElement("button");
    edit.textContent = "e";
    buttons.appendChild(edit);

    const mute = document.createElement("button");
    mute.textContent = "M";
    buttons.appendChild(mute);

    const solo = document.createElement("button");
    solo.textContent = "S";
    buttons.appendChild(solo);

    const indicator = document.createElement("div");
    indicator.classList.add("pattern-track-indicator");
    buttons.appendChild(indicator);

    buttons.appendChild(mute);
    buttons.appendChild(solo);

    this._canvas = document.createElement("canvas");
    this._canvas.height = 100;
    this._canvas.width = 1600;
    this._canvas.classList.add("pattern-track-pattern");

    this._canvas.addEventListener("pointerdown", () => {
      this.emit("select", this);
    });

    this.domElement.appendChild(instrumentPanel);
    this.domElement.appendChild(indicator);
    this.domElement.appendChild(this._canvas);
  }
}
