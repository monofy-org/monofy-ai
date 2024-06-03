import EventObject from "../../../elements/src/EventObject";
import { Instrument } from "../abstracts/Instrument";
import { FMBass } from "../plugins/fmbass/FMBass";
import { Sampler } from "../plugins/sampler/Sampler";
import { IPattern, IProject, ISequence, ITimelineSequence } from "../schema";
import { IInstrument } from "./IInstrument";
import { AudioClock } from "./components/AudioClock";

export class Project extends EventObject<"update"> implements IProject {
  title = "Untitled";
  description = "";
  tempo = 120;
  instruments: Instrument[] = [];
  patterns: IPattern[] = [];
  sequences: ISequence[] = [];
  timeline: ITimelineSequence[] = [];

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
      sequences: this.sequences,
      timeline: this.timeline,
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
    this.timeline = project.timeline;
    this.instruments = [];

    for (const instrument of project.instruments as IInstrument[]) {
      console.log("debug", instrument);
      if (instrument.id === "sampler") {
        this.instruments.push(new Sampler(this.audioClock));
      } else if (instrument.id === "fm_bass") {
        this.instruments.push(new FMBass(this.audioClock));
      } else {
        console.error("Unknown instrument", instrument);
      }
    }

    console.log("emitting update...");
    this.emit("update", { type: "project", value: this });
  }
}
