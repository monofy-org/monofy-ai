import { BaseElement } from "../../../../elements/src/elements/BaseElement";

export class MixerChannel extends BaseElement<"update"> {
  private readonly _channel: number;
  private readonly _volume: HTMLInputElement;
  private readonly _mute: HTMLInputElement;
  private readonly _solo: HTMLInputElement;
  private readonly _label: HTMLSpanElement;  

  get channel() {
    return this._channel;
  }

  get volume() {
    return this._volume.valueAsNumber;
  }

  set volume(value: number) {
    this._volume.value = value.toString();
  }

  get mute() {
    return this._mute.checked;
  }

  set mute(value: boolean) {
    this._mute.checked = value;
  }

  get solo() {
    return this._solo.checked;
  }

  set solo(value: boolean) {
    this._solo.checked = value;
  }

  constructor(channel: number, label: string, readonly gainNode: GainNode) {
    super("div", "mixer-channel");

    this._channel = channel;

    this._label = document.createElement("span");
    this._label.textContent = label;
    this.domElement.appendChild(this._label);

    const volume = document.createElement("input");
    volume.type = "range";
    volume.min = "0";
    volume.max = "1.2";
    volume.step = "0.01";
    volume.value = "1";

    const mute = document.createElement("input");
    mute.type = "checkbox";

    const solo = document.createElement("input");
    solo.type = "checkbox";

    this._volume = volume;
    this._mute = mute;
    this._solo = solo;

    this.domElement.appendChild(volume);
    this.domElement.appendChild(mute);
    this.domElement.appendChild(solo);

    volume.addEventListener("input", () => {
      this.emit("update");
    });

    mute.addEventListener("change", () => {
      this.emit("update");
    });

    solo.addEventListener("change", () => {
      this.emit("update");
    });
  }
}
