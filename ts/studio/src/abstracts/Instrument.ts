import type { Project } from "../elements/Project";
import type { ISourceEvent } from "../elements/components/SamplerSlot";
import { Plugin } from "../plugins/plugins";
import { IInstrumentSettings } from "../schema";

export interface IInstrument extends IInstrumentSettings {  

  trigger(
    note: number,
    when: number,
    velocity: number
  ): ISourceEvent | void;

  release(note: number, when: number): void;
}

export abstract class Instrument extends Plugin implements IInstrument {
  
  inputPort?: number | undefined;
  inputChannel?: number | undefined;
  transpose = 0;
  output: GainNode;

  constructor(project: Project, public mixerChannel = 0) {
    super(project);
    this.output = project.audioClock.audioContext.createGain();
    this.output.connect(project.mixer.channels[mixerChannel].gainNode);
  }

  abstract trigger(
    note: number,
    when: number,
    velocity?: number
  ): ISourceEvent | void;

  abstract release(note: number, when: number): void;

}
