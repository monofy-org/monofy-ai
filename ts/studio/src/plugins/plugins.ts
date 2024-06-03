import { EventDataMap } from "../../../elements/src/EventObject";
import { BaseElement } from "../../../elements/src/elements/BaseElement";
import { DraggableWindow } from "../../../elements/src/elements/DraggableWindow";
import { AudioClock } from "../elements/components/AudioClock";
import { ControllerGroup, IHasControls } from "../schema";

export abstract class Plugin
  extends BaseElement<"update">
  implements IHasControls
{
  abstract name: string;
  abstract version: string;
  abstract description: string;
  abstract author: string;
  abstract controllerGroups: ControllerGroup[];
  abstract window: DraggableWindow<"update" | keyof EventDataMap>;

  constructor(readonly audioClock: AudioClock) {
    super("div", "plugin-container");
  }
}

export abstract class Plugins {
  static readonly plugins: (typeof Plugin)[] = [];

  static register(plugin: typeof Plugin) {
    if (Plugins.plugins.includes(plugin)) {
      console.warn("Plugin already registered:", plugin);
      return;
    }
    Plugins.plugins.push(plugin);
  }

  static get(name: string) {
    return this.plugins.find((plugin) => plugin.name === name);
  }
}
