import type { EventDataMap } from "../../../elements/src/EventObject";
import { BaseElement } from "../../../elements/src/elements/BaseElement";
import type { DraggableWindow } from "../../../elements/src/elements/DraggableWindow";
import type { Mixer } from "../elements/Mixer";
import type { AudioClock } from "../elements/components/AudioClock";
import type { ControllerGroup, IHasControls } from "../schema";

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
  abstract window?: DraggableWindow<"update" | keyof EventDataMap>;

  constructor(
    readonly audioClock: AudioClock,
    readonly mixer: Mixer
  ) {
    console.assert(audioClock, "audioClock is required");
    console.assert(mixer, "mixer is required");
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

  static instantiate<T extends Plugin>(
    id: string,
    audioClock: AudioClock,
    mixer: Mixer
  ) {
    return new (this.get(id) as any)(audioClock, mixer) as T;
  }
}
