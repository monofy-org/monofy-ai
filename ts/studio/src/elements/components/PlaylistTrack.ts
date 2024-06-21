import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { ITrackOptions } from "../../schema";

export class PlaylistTrack
  extends BaseElement<"update">
  implements ITrackOptions
{
  private _track: ITrackOptions;

  get mute() {
    return this._track.mute;
  }

  get solo() {
    return this._track.solo;
  }

  get selected() {
    return false;
  }

  get name() {
    return this._track.name;
  }

  constructor(track: ITrackOptions) {
    super("div", "playlist-track");

    this._track = track;

    const settings = document.createElement("div");
    settings.classList.add("playlist-track-panel");

    const labelAndButtons = document.createElement("div");
    labelAndButtons.classList.add("playlist-track-label-and-buttons");

    const label = document.createElement("div");    
    label.textContent = track.name;

    const buttons = document.createElement("div");
    buttons.classList.add("playlist-track-buttons");    

    labelAndButtons.appendChild(label);
    labelAndButtons.appendChild(buttons);
    settings.appendChild(labelAndButtons);

    const mute = document.createElement("div");
    mute.classList.add("track-button");
    mute.classList.add("track-mute-button");
    mute.textContent = "M";
    mute.addEventListener("pointerdown", () => {
      mute.classList.toggle("active");
    });
    buttons.appendChild(mute);

    const solo = document.createElement("div");
    solo.classList.add("track-button");
    solo.classList.add("track-solo-button");
    solo.textContent = "S";
    solo.addEventListener("pointerdown", () => {
      solo.classList.toggle("active");
    });
    buttons.appendChild(solo);

    this.domElement.appendChild(settings);
  }

  trigger(note: number, velocity: number) {
    console.log("Trigger", note, velocity);
  }

  release(note: number) {
    console.log("Release", note);
  }
}
