import { EventDataMap } from "../../../elements/src/EventObject";
import { BaseElement } from "../../../elements/src/elements/BaseElement";
import { DraggableWindow } from "../../../elements/src/elements/DraggableWindow";
import { AudioClock } from "../elements/components/AudioClock";
import { ControllerGroup, IHasControls } from "../schema";

export abstract class Plugin
  extends BaseElement<"update">
  implements IHasControls
{
  abstract id: string;
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

export class Plugins {
  private static readonly _plugins: { [key: string]: typeof Plugin } = {};

  static register(name: string, plugin: typeof Plugin) {
    if (this._plugins[name]) {
      console.warn("Plugin already registered:", plugin);
      return;
    }
    this._plugins[name] = plugin;
  }

  static get<T extends Plugin>(name: string): T {
    return this._plugins[name] as any as T;
  }

  static instantiate<T extends Plugin>(id: string, audioClock: AudioClock) {
    return new (this.get(id) as any)(audioClock) as T;
  }
}
