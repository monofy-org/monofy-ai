import {
  DraggableWindow,
  IWindowOptions,
} from "../../../elements/src/elements/DraggableWindow";
import { Mixer } from "../elements/Mixer";
import type { ProjectUI } from "../elements/ProjectUI";
import { Effect } from "../elements/components/Effect";

export interface IEffectWindowOptions extends IWindowOptions {
  mixer: Mixer;
  mixerChannel: number;
  effect: Effect;
}

export class EffectWindow extends DraggableWindow {
  private _settingsBar: HTMLDivElement;

  get mixerChannel() {
    return this.options.mixerChannel;
  }

  set mixerChannel(value: number) {
    this.options.mixerChannel = value;
  }

  constructor(
    ui: ProjectUI,
    readonly options: IEffectWindowOptions
  ) {
    super(ui.container, options);

    this._settingsBar = document.createElement("div");
    this._settingsBar.classList.add("window-settings-bar");

    this.domElement.insertBefore(this._settingsBar, this.content);
  }
}
