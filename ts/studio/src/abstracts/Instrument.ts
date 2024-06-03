import { IInstrument } from "../elements/IInstrument";
import { AudioClock } from "../elements/components/AudioClock";
import { ISourceEvent } from "../elements/components/SamplerSlot";
import { Plugin } from "../plugins/plugins";

export abstract class Instrument extends Plugin implements IInstrument {

  abstract id: string;

  constructor(readonly audioClock: AudioClock) {
    super(audioClock);
  }

  abstract trigger(
    note: number,
    when: number,
    velocity: number | undefined
  ): ISourceEvent | undefined;
  abstract release(note: number): void;
}
