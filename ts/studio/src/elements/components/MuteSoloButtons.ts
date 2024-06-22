import { BaseElement } from "../../../../elements/src/elements/BaseElement";

export class MuteSoloButtons extends BaseElement<"change"> {
  private _mute: boolean = false;
  private _solo: boolean = false;
  private _muteButton: HTMLDivElement;
  private _soloButton: HTMLDivElement;

  get mute() {
    return this._mute;
  }

  set mute(value: boolean) {
    this._mute = value;
    this._muteButton.classList.toggle("active", this._mute);
  }

  get solo() {
    return this._solo;
  }

  set solo(value: boolean) {
    this._solo = value;
    this._soloButton.classList.toggle("active", this._solo);
  }

  constructor() {
    super("div", "track-buttons");

    const mute = document.createElement("div");
    this._muteButton = mute;
    mute.classList.add("track-button");
    mute.classList.add("track-mute-button");
    mute.textContent = "M";
    mute.addEventListener("pointerdown", () => {
      this._mute = !this._mute;
      mute.classList.toggle("active", this._mute);
      this.emit("change");
    });
    this.domElement.appendChild(mute);

    const solo = document.createElement("div");
    this._soloButton = solo;
    solo.classList.add("track-button");
    solo.classList.add("track-solo-button");
    solo.textContent = "S";
    solo.addEventListener("pointerdown", () => {
      this._solo = !this._solo;
      solo.classList.toggle("active", this._solo);
      this.emit("change");
    });
    this.domElement.appendChild(solo);
  }
}
