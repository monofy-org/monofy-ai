import EventObject from "../../../elements/src/EventObject";
import { Instrument } from "../abstracts/Instrument";
import { FMBass } from "../plugins/fmbass/FMBass";
import { Sampler } from "../plugins/sampler/Sampler";
import { IPattern, IProject, ISequence, ITimelineSequence } from "../schema";
import { IInstrument } from "./IInstrument";
import { AudioClock } from "./components/AudioClock";

export class Project extends EventObject<"update"> implements IProject {
  title: string;
  description: string;
  tempo: number;
  instruments: Instrument[] = [];
  patterns: IPattern[] = [];
  sequences: ISequence[] = [];
  timeline: ITimelineSequence[] = [];

  constructor(
    readonly audioClock: AudioClock,
    title = "Untitled",
    description = ""
  ) {
    super();
    this.title = title;
    this.description = description;
    this.tempo = 120;
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

    const project = JSON.parse(data);
    this.title = project.title;
    this.description = project.description;
    this.tempo = project.tempo;
    this.patterns = project.patterns;
    this.sequences = project.sequences;
    this.timeline = project.timeline;
    this.instruments = [];

    for (const instrument of project.instruments) {
      console.log("debug", instrument);
      const inst = instrument as IInstrument;
      if (inst.id === "sampler") {
        this.instruments.push(new Sampler(this.audioClock));
      } else if (inst.id === "fm_bass") {
        this.instruments.push(new FMBass(this.audioClock));
      }

    }

    // TODO error checking

    this.emit("update", { type: "project", value: this });
  }
}
