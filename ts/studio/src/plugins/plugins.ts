import { InstrumentWindow } from "../abstracts/InstrumentWindow";
import type { Project } from "../elements/Project";

export abstract class Plugin {
  abstract id: string;
  abstract name: string;
  abstract version: string;
  abstract description: string;
  abstract author: string;  
  private readonly _project;

  abstract Window: typeof InstrumentWindow;

  get project() {
    return this._project;
  }

  constructor(project: Project) {        
    console.assert(project.audioClock, "audioClock is required");
    console.assert(project.mixer, "mixer is required");
    this._project = project;
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

  static instantiate<T extends Plugin>(project: Project, id: string) {
    return new (this.get(id) as any)(project) as T;
  }
}
