import EventObject from "../../../elements/src/EventObject";
import { Instrument } from "../abstracts/Instrument";
import { Plugins } from "../plugins/plugins";
import { IPattern, IPlaylist, IProject, ITrackOptions } from "../schema";
import { IInstrument } from "./IInstrument";
import { AudioClock } from "./components/AudioClock";

export class Project extends EventObject<"update"> implements IProject {
  title = "Untitled";
  description = "";
  tempo = 120;
  instruments: Instrument[] = [];
  patterns: IPattern[] = [];
  tracks: ITrackOptions[] = [];
  playlist: IPlaylist = { events: [] };

  constructor(
    readonly audioClock: AudioClock,
    project?: IProject
  ) {
    super();
    if (project) {
      setTimeout(() => {
        this.load(project);
      }, 0);
    }

    audioClock.on("start", () => {
      console.log("DEBUG START", this.playlist);
      for (const event of this.playlist.events) {        
        // const e = event.value as IPattern;
        console.log("DEBUG ITEM", event.start, event);
      }
    });
  }

  serialize(): string {
    return JSON.stringify({
      title: this.title,
      description: this.description,
      tempo: this.tempo,
      instruments: this.instruments.map((instrument) => {
        return {
          name: instrument.name,
          controllerGroups: instrument.controllerGroups,
        };
      }),
      patterns: this.patterns,
      timeline: this.playlist,
    });
  }

  deserialize(data: string): void {
    console.log("Project deserialize", data);
    const project = JSON.parse(data) as IProject;

    // TODO error checking

    this.load(project);
  }

  load(project: IProject): void {
    console.log("Project load", project);

    this.title = project.title;
    this.description = project.description;
    this.tempo = project.tempo;
    this.patterns = project.patterns;
    this.playlist = project.playlist;
    this.instruments = [];

    for (const instrument of project.instruments as IInstrument[]) {
      console.log("loading instrument", instrument);
      const T = Plugins.get(instrument.id);
      const instance: typeof T = Plugins.instantiate<typeof T>(
        instrument.id,
        this.audioClock
      );
      this.instruments.push(instance as Instrument);
    }

    console.log("emitting update...");
    this.emit("update", { type: "project", value: this });
  }
}
