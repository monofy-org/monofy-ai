import { Instrument } from "../../abstracts/Instrument";
import type { Project } from "../../elements/Project";
import { MultisamplerWindow } from "./MultisamplerWindow";
import { Sampler } from "./Sampler";

export class Multisampler extends Instrument {
  readonly name = "Multisampler";
  readonly version = "0.0.1";
  readonly description = "A simple multisampler";
  readonly author = "Johnny Street";
  readonly id = "multisampler";

  readonly Window = MultisamplerWindow;

  samplers: Sampler[] = [];  

  constructor(project: Project) {
    super(project);

    this.transpose = -24;
  }

  trigger(note: number, when: number, velocity: number) {
    note += this.transpose;
    console.log("Sampler trigger", note);
    const slot = this.samplers[note];
    if (slot) {
      for (const sampler of this.samplers) {
        if (sampler.cutByGroups.includes(slot.cutGroup)) {
          sampler.release(note, when);
        }
      }
      return slot.trigger(note, when, velocity);
    }
  }

  release(note: number, when = 0): void {
    note += this.transpose;
    if (this.samplers[note]) {
      this.samplers[note].release(note, when);
    }
  }
}
