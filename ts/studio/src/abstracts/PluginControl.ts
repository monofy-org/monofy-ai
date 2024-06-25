import { BaseElement } from "../../../elements/src/elements/BaseElement";
import { ControllerValue, IPluginControl } from "../schema";

export abstract class PluginControl
  extends BaseElement<"change">
{
  protected _label: HTMLSpanElement;

  abstract value: ControllerValue;

  constructor(readonly settings: IPluginControl) {
    super("div", "control");
    this._label = document.createElement("span");
    this._label.classList.add("label");
    this._label.textContent = settings.label;
    this.domElement.appendChild(this._label);
  }
}
