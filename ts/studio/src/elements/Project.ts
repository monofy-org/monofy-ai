import EventObject from "../../../elements/src/EventObject";
import { TreeViewItem } from "../../../elements/src/elements/TreeView";
import type { IInstrument, Instrument } from "../abstracts/Instrument";
import { Plugins } from "../plugins/plugins";
import { IAudioItem, IEvent, IPattern, IPlaylist, IProject } from "../schema";
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
  private readonly _sources: AudioBufferSourceNode[] = [];
  readonly mixer: Mixer;

  constructor(audioClock: AudioClock, project?: IProject) {
    console.assert(audioClock, "audioClock is required");

    super();

    this.audioClock = audioClock;

    this.mixer = new Mixer(audioClock.audioContext);

    audioClock.on("start", () => {
      console.log("DEBUG START", this.playlist);
      for (const e of this.playlist.events) {
        if (!(e.value instanceof TreeViewItem)) {
          throw new Error("Invalid event value");
        }

        const event = e.value as TreeViewItem;

        console.log("DEBUG ITEM", event);

        if (event.type === "audio") {
          this._queueAudioEvent(e);
        } else if (event.type === "pattern") {
          console.log("DEBUG PATTERN", event.item.data as IPattern);
        } else {
          console.error("UNKNOWN EVENT", event);
          console.error("EVENT DATA", event.item.data);
        }
      }
    });

    audioClock.on("stop", () => {
      for (const source of this._sources) {
        source.stop();
      }
      this._sources.length = 0;
    });

    if (project) {
      setTimeout(() => {
        this.load(project);
      }, 0);
    }
  }

  _queueAudioEvent(event: IEvent) {
    const source = this.audioClock.audioContext.createBufferSource();
    const item = event.value as TreeViewItem;
    const audioItem = item.data as IAudioItem;
    source.buffer = audioItem.buffer;
    source.connect(this.mixer.channels[0].gainNode);
    source.start(this.audioClock.getBeatTime(event.start));
    source.stop(this.audioClock.getBeatTime(event.start + event.duration));
    this._sources.push(source);
    source.onended = () => {
      const index = this._sources.indexOf(source);
      if (index >= 0) {
        this._sources.splice(index, 1);
      }
    };
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
