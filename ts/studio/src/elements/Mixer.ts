import { MathHelpers } from "../abstracts/MathHelpers";
import { IMixer, IMixerChannel } from "../schema";

export class MixerChannel implements IMixerChannel {
  gainNode: GainNode;
  mute = false;
  solo = false;

  get gain() {
    return this.gainNode.gain.value;
  }

  constructor(
    readonly audioContext: AudioContext,
    public label: string,
    gain = 1,
    readonly outputs: number[] = [0]
  ) {
    this.gainNode = audioContext.createGain();
    this.gainNode.gain.value = MathHelpers.linearToGain(gain);
  }
}

export class Mixer implements IMixer {
  channels: MixerChannel[] = [];

  constructor(readonly audioContext: AudioContext) {
    const master = new MixerChannel(audioContext, "Master", 1);
    master.gainNode.connect(audioContext.destination);
    this.channels.push(master);
    for (let i = 1; i <= 16; i++) {
      const channel = new MixerChannel(audioContext, i.toString());
      channel.gainNode.connect(master.gainNode);
      this.channels.push(channel);
    }
  }
}
