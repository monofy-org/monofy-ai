import EventObject from "../../../elements/src/EventObject";
import type { IInstrument, Instrument } from "../abstracts/Instrument";
import { Plugins } from "../plugins/plugins";
import { IPattern, IPlaylist, IProject } from "../schema";
import { Mixer } from "./Mixer";
import { AudioClock } from "./components/AudioClock";

export interface IProjectUpdateEvent {
  type: "project" | "pattern";
  value: Project | IPattern;
}

export class Project extends EventObject<"update"> implements IProject {
  title = "Untitled";
  description = "";
  tempo = 120;
  instruments: Instrument[] = [];
  patterns: IPattern[] = [];
  playlist: IPlaylist = { tracks: [], events: [] };
  audioClock: AudioClock;
  readonly mixer: Mixer;

  constructor(audioClock: AudioClock, project?: IProject) {
    console.assert(audioClock, "audioClock is required");

    super();

    this.audioClock = audioClock;

    this.mixer = new Mixer(audioClock.audioContext);

    audioClock.on("start", () => {
      console.log("DEBUG START", this.playlist);
      for (const event of this.playlist.events) {
        // const e = event.value as IPattern;
        console.log("DEBUG ITEM", event.start, event);
        if (event.value instanceof AudioBuffer) {
          const source = audioClock.audioContext.createBufferSource();
          source.buffer = event.value;
          source.connect(this.mixer.channels[0].gainNode);
          source.start(audioClock.getBeatTime(event.start));
          source.stop(audioClock.getBeatTime(event.start + event.duration));
          console.log("DEBUG PLAY", audioClock.getBeatTime(event.start));
        } else {
          console.error("DEBUG VALUE", event.value);        
        }
      }
    });

    if (project) {
      setTimeout(() => {
        this.load(project);
      }, 0);
    }
  }

  serialize(): string {
    return JSON.stringify({
      title: this.title,
      description: this.description,
      tempo: this.tempo,
      instruments: this.instruments.map((instrument) => {
        return {
          name: instrument.name,
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
        this,
        instrument.id
      );
      this.instruments.push(instance as Instrument);
    }

    console.log("emitting update...");
    this.emit("update", { type: "project", value: this });
  }
}
