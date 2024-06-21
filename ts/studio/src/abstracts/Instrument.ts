import { IInstrument } from "../elements/IInstrument";
import { Mixer } from "../elements/Mixer";
import { AudioClock } from "../elements/components/AudioClock";
import { ISourceEvent } from "../elements/components/SamplerSlot";
import { Plugin } from "../plugins/plugins";

export abstract class Instrument extends Plugin implements IInstrument {
  inputPort?: number | undefined;
  inputChannel?: number | undefined;
  output: GainNode;

  constructor(audioClock: AudioClock, mixer: Mixer, mixerChannel = 0) {
    super(audioClock, mixer);
    this.output = audioClock.audioContext.createGain();
    this.output.connect(mixer.channels[mixerChannel].gainNode);
  }

  abstract trigger(
    note: number,
    when: number,
    velocity?: number
  ): ISourceEvent | undefined;

  abstract release(note: number, when: number): void;
}
