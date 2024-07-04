import { DraggableNumber } from "../../../elements/src/elements/DraggableNumber";

import {
  DraggableWindow,
  IWindowOptions,
} from "../../../elements/src/elements/DraggableWindow";
import type { ProjectUI } from "../elements/ProjectUI";
import { Instrument } from "./Instrument";

export interface IInstrumentWindowOptions extends IWindowOptions {
  mixerChannel: number;
  instrument: Instrument;
}

export class InstrumentWindow extends DraggableWindow {
  private _settingsBar: HTMLDivElement;

  constructor(
    ui: ProjectUI,
    readonly instrument: Instrument
  ) {
    super({
      title: instrument.name,
      width: 400,
      height: 300,
      persistent: true,
    });

    const mixerChannelSelector = new DraggableNumber(
      instrument.mixerChannel,
      0,
      16
    );
    mixerChannelSelector.on("change", (value) => {
      this.instrument.mixerChannel = value as number;
      console.assert(ui.project.mixer, "instrument.mixer is undefined!");
      instrument.output.disconnect();
      instrument.output.connect(
        ui.project.mixer.channels[this.instrument.mixerChannel].gainNode
      );
    });

    this._settingsBar = document.createElement("div");
    this._settingsBar.classList.add("window-settings-bar");
    this._settingsBar.appendChild(mixerChannelSelector.domElement);

    this.domElement.insertBefore(this._settingsBar, this.content);

    ui.container.addWindow(this);
  }
}
