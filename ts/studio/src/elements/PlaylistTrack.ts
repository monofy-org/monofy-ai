import { BaseElement } from "../../../elements/src/elements/BaseElement";

export class PlaylistTrack extends BaseElement<"update"> {
  constructor(name: string) {
    super("div", "playlist-track");

    const settings = document.createElement("div");
    settings.classList.add("instrument-panel");

    const label = document.createElement("div");
    settings.appendChild(label);
    label.textContent = name;

    const buttons = document.createElement("div");
    buttons.classList.add("playlist-track-selector");
    settings.appendChild(buttons);

    const mute = document.createElement("button");
    mute.textContent = "M";
    buttons.appendChild(mute);

    const solo = document.createElement("button");
    solo.textContent = "S";
    buttons.appendChild(solo);

    const track = document.createElement("div");
    track.classList.add("pattern-panel");

    this.domElement.appendChild(settings);
    this.domElement.appendChild(track);
  }
}
