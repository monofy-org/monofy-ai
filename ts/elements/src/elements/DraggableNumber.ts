import { BaseElement } from "./BaseElement";

export class DraggableNumber extends BaseElement<"change"> {
  private _value: number;
  private _step: number;
  private _hideZero: string | null;

  get value() {
    return this._value;
  }

  set value(value: number) {
    this._value = value;
    if (value === 0 && this._hideZero !== null) {
      this.domElement.textContent = this._hideZero;
    } else {
      this.domElement.textContent = value.toString();
    }
  }

  constructor(
    value: number,
    min: number,
    max: number,
    step = 1,
    sensitivity = 0.25,
    hideZero: string | null = "-"
  ) {
    super("div", "draggable-number");

    this._hideZero = hideZero;
    this._value = value;
    this._step = step;

    this.domElement.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      const startY = e.clientY;
      const startValue = this.value;
      const onMouseMove = (event: PointerEvent) => {
        const oldValue = this.value;
        const dy = Math.round(-(event.clientY - startY) * sensitivity);
        const newValue = startValue + dy * this._step;
        this.value = Math.max(min, Math.min(max, newValue));
        if (oldValue !== this.value) this.emit("change", this.value);
      };
      const onMouseUp = () => {
        document.removeEventListener("pointermove", onMouseMove);
        document.removeEventListener("pointerup", onMouseUp);
      };
      document.addEventListener("pointermove", onMouseMove);
      document.addEventListener("pointerup", onMouseUp);
    });

    this.value = value || 0;
  }
}
