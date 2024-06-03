import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { ControlType, IPluginControl } from "../../schema";

export class Slider extends BaseElement<"update"> implements IPluginControl {
  private readonly _name: string;
  private readonly _label: string;
  private readonly _type: ControlType = "slider";
  private readonly _default: number;
  private readonly _min: number;
  private readonly _max: number;
  private readonly _step: number;
  private readonly _range: HTMLInputElement;
  private _value: number;

  get name() {
    return this._name;
  }

  get label() {
    return this._label;
  }

  get type() {
    return this._type;
  }

  get value() {
    return this._value;
  }

  get default() {
    return this._default;
  }

  get min() {
    return this._min;
  }

  get max() {
    return this._max;
  }

  get step() {
    return this._step;
  }

  set value(value: number) {
    this._value = value;
    this._range.value = value.toString();
  }

  get element() {
    return this._range;
  }

  constructor(
    name: string,
    label: string,
    min: number,
    max: number,
    step: number,
    value: number
  ) {
    super("div", "slider");

    this._name = name;
    this._label = label;
    this._min = min;
    this._max = max;
    this._step = step;
    this._value = value;
    this._default = value;

    this._range = document.createElement("input");
    this._range.type = "range";
    this._range.min = min.toString();
    this._range.max = max.toString();
    this._range.step = step.toString();
    this._range.value = value.toString();
    this._range.style.transform = "rotate(270deg)";

    this._range.addEventListener("input", () => {
      this._value = parseFloat(this._range.value);
    });

    this.domElement.appendChild(this._range);
  }
}
