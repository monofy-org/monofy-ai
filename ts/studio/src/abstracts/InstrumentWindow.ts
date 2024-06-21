import { DraggableNumber } from "../../../elements/src/elements/DraggableNumber";

import {
  DraggableWindow,
  IWindowOptions,
} from "../../../elements/src/elements/DraggableWindow";
import { Instrument } from "./Instrument";

export interface IInstrumentWindowOptions extends IWindowOptions {
  mixerChannel: number;
}

export class InstrumentWindow extends DraggableWindow {
  private _settingsBar: HTMLDivElement;

  get mixerChannel() {
    return this.options.mixerChannel;
  }

  set mixerChannel(value: number) {
    this.options.mixerChannel = value;
  }

  constructor(
    readonly options: IInstrumentWindowOptions,
    readonly instrument: Instrument,    
  ) {
    super(options);

    const mixerChannelSelector = new DraggableNumber(
      options.mixerChannel,
      0,
      16
    );
    mixerChannelSelector.on("change", (value) => {
      this.mixerChannel = value as number;
      console.assert(instrument.mixer, "instrument.mixer is undefined!");
      instrument.output.disconnect();
      instrument.output.connect(
        instrument.mixer.channels[this.mixerChannel].gainNode
      );
    });

    this._settingsBar = document.createElement("div");
    this._settingsBar.classList.add("window-settings-bar");
    this._settingsBar.appendChild(mixerChannelSelector.domElement);

    this.domElement.insertBefore(this._settingsBar, this.content);
  }
}
