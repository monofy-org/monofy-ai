import { SelectableElement } from "../../../../elements/src/elements/SelectableElement";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import { IMixerChannel } from "../../schema";
import { Effect } from "./Effect";
import { MuteSoloButtons } from "./MuteSoloButtons";

export class MixerChannelStrip extends SelectableElement<"change"> {
  private readonly _channel: IMixerChannel;
  private readonly _volume: HTMLInputElement;
  private readonly _label: HTMLSpanElement;
  readonly muteSoloButtons: MuteSoloButtons;
  readonly effects: Effect[] = [];

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
    return this.muteSoloButtons.mute;
  }

  set mute(value: boolean) {
    if (value !== this.muteSoloButtons.mute) {
      this.muteSoloButtons.mute = value;
      this.channel.mute = value;
    }
  }

  get solo() {
    return this.muteSoloButtons.solo;
  }

  set solo(value: boolean) {
    if (value !== this.muteSoloButtons.solo) {
      this.muteSoloButtons.solo = value;
      this.channel.solo = value;
    }
  }

  constructor(
    group: SelectableGroup,
    channel: IMixerChannel,
    readonly isMaster = false
  ) {
    super(group, "div", "mixer-channel");

    this._channel = channel;

    this._label = document.createElement("span");
    this._label.textContent = channel.label;
    this.domElement.appendChild(this._label);

    const volume = document.createElement("input");
    volume.type = "range";
    volume.min = "0";
    volume.max = "1.2";
    volume.step = "0.01";
    volume.value = "1";

    this._volume = volume;

    this.domElement.appendChild(volume);

    this.muteSoloButtons = new MuteSoloButtons();
    this.domElement.appendChild(this.muteSoloButtons.domElement);

    volume.addEventListener("input", () => {
      this.emit("change");
    });

    this.muteSoloButtons.on("change", () => {
      this.emit("change");
    });
  }
}
