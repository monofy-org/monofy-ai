import { BaseElement } from "../../../elements/src/elements/BaseElement";
import { IGridItem } from "./Grid";

export class PatternTrack extends BaseElement<"update"> {
  constructor(
    name: string,
    readonly pattern: IGridItem[]
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

    const patternPanel = document.createElement("div");
    patternPanel.classList.add("pattern-track-pattern");

    this.domElement.appendChild(instrumentPanel);
    this.domElement.appendChild(indicator);
    this.domElement.appendChild(patternPanel);
  }
}
