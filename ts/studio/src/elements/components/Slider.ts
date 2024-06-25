import { PluginControl } from "../../abstracts/PluginControl";
import { IPluginControl } from "../../schema";

export class Slider extends PluginControl {
  private readonly _range: HTMLInputElement;

  get value() {
    return this.settings.value as number;
  }

  set value(value: number) {
    this.settings.value = value;
    this._range.value = value.toString();
  }

  get element() {
    return this._range;
  }

  constructor(options: IPluginControl) {
    super({ ...options, controlType: "slider" });

    this._range = document.createElement("input");
    this._range.type = "range";
    this._range.min = options.min.toString();
    this._range.max = options.max.toString();
    this._range.step = options.step.toString();
    this._range.value = options.value.toString();

    this._range.addEventListener("input", () => {
      this.settings.value = parseFloat(this._range.value);
      this.emit("change", this);
    });

    this._range.addEventListener("change", () => {
      this._range.blur();
    });

    this.domElement.appendChild(this._range);
  }
}
