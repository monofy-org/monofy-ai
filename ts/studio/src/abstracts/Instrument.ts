import { IInstrument } from "../elements/IInstrument";
import { ISourceEvent } from "../elements/components/SamplerSlot";
import { Plugin } from "../plugins/plugins";

export abstract class Instrument extends Plugin implements IInstrument {
  inputPort?: number | undefined;
  inputChannel?: number | undefined;

  abstract trigger(
    note: number,
    when: number,
    velocity?: number
  ): ISourceEvent | undefined;

  abstract release(note: number, when: number): void;
}
