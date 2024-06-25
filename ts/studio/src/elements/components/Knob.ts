import { PluginControl } from "../../abstracts/PluginControl";
import { IPluginControl } from "../../schema";

export class Knob extends PluginControl {
  private _value = 0;
  private _handle: HTMLDivElement;
  public sensitivity = 0.25;

  get value() {
    return this._value;
  }

  constructor(readonly settings: IPluginControl) {
    super({
      ...settings,
      controlType: "knob",
    });

    this.domElement.classList.add("knob");

    const indicator = document.createElement("div");
    indicator.classList.add("knob-indicator");

    this.sensitivity = (settings.max - settings.min / settings.step) * 0.01;

    this._handle = document.createElement("div");
    this._handle.appendChild(indicator);
    this._handle.classList.add("knob-handle");
    this._handle.addEventListener(
      "pointerdown",
      this._onPointerDown.bind(this)
    );
    this.domElement.insertBefore(this._handle, this._label);

    this._setValue(settings.value as number);
  }

  private _onPointerDown(e: PointerEvent) {
    e.preventDefault();
    e.stopPropagation();

    const startValue = this._value;
    const startPosition = { x: e.clientX, y: e.clientY };

    const onmove = (e: PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const offset = -(e.clientY - startPosition.y);
      const step = e.ctrlKey ? this.settings.step * 0.01 : 1;
      let value = startValue + offset * this.sensitivity * step;
      value -= value % step;
      this._setValue(value);
      this.emit("change", this._value);
      return false;
    };

    const onrelease = (e: PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();
      window.removeEventListener("pointermove", onmove);
      window.removeEventListener("pointerup", onrelease);
      this._handle.releasePointerCapture(e.pointerId);
      this._handle.classList.remove("active");
      return false;
    };

    window.addEventListener("pointermove", onmove);
    window.addEventListener("pointerup", onrelease);

    // this._handle.setPointerCapture(event.pointerId);
    // this._handle.focus();
    this.emit("change", this._value);
    this._handle.classList.add("active");
    return false;
  }

  private _setValue(value: number) {
    this._value = Math.min(
      this.settings.max,
      Math.max(this.settings.min, value)
    );
    this._handle.style.transform = `rotate(${
      (this._value * 280) / (this.settings.max - this.settings.min) - 50
    }deg)`;
  }
}
