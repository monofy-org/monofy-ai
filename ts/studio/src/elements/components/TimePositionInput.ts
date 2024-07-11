import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { DraggableNumber } from "../../../../elements/src/elements/DraggableNumber";

export class TimePositionInput extends BaseElement<"change"> {
  private _inputHours: DraggableNumber;
  private _inputMinutes: DraggableNumber;
  private _inputSeconds: DraggableNumber;
  private _inputMillis: DraggableNumber;

  get value() {
    return (
      this._inputHours.value * 3600 +
      this._inputMinutes.value * 60 +
      this._inputSeconds.value +
      this._inputMillis.value / 1000
    );
  }

  set value(value: number) {
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const seconds = Math.floor(value % 60);
    const milliseconds = Math.floor((value % 1) * 1000);

    this._inputHours.value = hours;
    this._inputMinutes.value = minutes;
    this._inputSeconds.value = seconds;
    this._inputMillis.value = milliseconds;
  }

  constructor() {
    super("div", "time-position-input");

    this._inputHours = new DraggableNumber(0, 0, 99, 1, undefined, null, 2);
    this._inputHours.on("change", () => this.emit("change"));
    this.domElement.appendChild(this._inputHours.domElement);

    const hoursLabel = document.createElement("label");
    hoursLabel.textContent = "h";
    this.domElement.appendChild(hoursLabel);

    this._inputMinutes = new DraggableNumber(0, 0, 59, 1, undefined, null, 2);
    this._inputMinutes.on("change", () => this.emit("change"));
    this.domElement.appendChild(this._inputMinutes.domElement);

    const minutesLabel = document.createElement("label");
    minutesLabel.textContent = "m";
    this.domElement.appendChild(minutesLabel);

    this._inputSeconds = new DraggableNumber(0, 0, 59, 1, undefined, null, 2);
    this._inputSeconds.on("change", () => this.emit("change"));
    this.domElement.appendChild(this._inputSeconds.domElement);

    const millisecondsLabel = document.createElement("label");
    millisecondsLabel.textContent = ".";
    this.domElement.appendChild(millisecondsLabel);

    this._inputMillis = new DraggableNumber(0, 0, 999, 1, undefined, null, 3);
    this._inputMillis.on("change", () => this.emit("change"));
    this.domElement.appendChild(this._inputMillis.domElement);

    const secondsLabel = document.createElement("label");
    secondsLabel.textContent = "s";
    this.domElement.appendChild(secondsLabel);

    this.value = 0;
  }
}
