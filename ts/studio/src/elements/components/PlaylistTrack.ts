import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { ITrackOptions } from "../../schema";
import { MuteSoloButtons } from "./MuteSoloButtons";

export class PlaylistTrack
  extends BaseElement<"update">
  implements ITrackOptions
{
  private _track: ITrackOptions;
  readonly muteSoloButtons: MuteSoloButtons;

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

    this.muteSoloButtons = new MuteSoloButtons();
    this.muteSoloButtons.on("change", () => {
      this._track.mute = this.muteSoloButtons.mute;
      this._track.solo = this.muteSoloButtons.solo;
    });

    labelAndButtons.appendChild(label);
    labelAndButtons.appendChild(this.muteSoloButtons.domElement);
    settings.appendChild(labelAndButtons);
    this.domElement.appendChild(settings);
  }

  trigger(note: number, velocity: number) {
    console.log("Trigger", note, velocity);
  }

  release(note: number) {
    console.log("Release", note);
  }
}
