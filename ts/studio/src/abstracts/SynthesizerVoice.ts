import { AudioClock } from "../elements/components/AudioClock";
import { ISourceEvent } from "../elements/components/SamplerSlot";
import { ISynthesizerVoiceSettings } from "../schema";
import { IInstrument } from "./Instrument";

export abstract class SynthesizerVoice implements ISynthesizerVoiceSettings, IInstrument {
  _settings: ISynthesizerVoiceSettings | null = null;

  abstract name: string;
  abstract id: string;

  get settings(): ISynthesizerVoiceSettings {
    if (!this.settings) throw new Error("Voice not initialized");
    return this.settings;
  }

  private readonly _gain: GainNode;
  private _note: number | null = null;
  private _frequency: number | null = null;

  get gain() {
    return this._gain.gain.value;
  }

  get output() {
    return this._gain;
  }

  get note() {
    return this._note;
  }

  get lfoNumber() {
    return this.settings.lfoNumber;
  }

  get envelopeNumber() {
    return this.settings.envelopeNumber;
  }

  get frequency() {
    return this._frequency;
  }

  get detune() {
    return this.settings.detune;
  }

  get oscillators() {
    return this.settings.oscillators;
  }

  constructor(readonly audioClock: AudioClock) {
    this._gain = audioClock.audioContext.createGain();
  }

  loadSettings(settings: ISynthesizerVoiceSettings) {
    this._settings = settings;
  }

  abstract trigger(
    note: number,
    velocity: number,
    when: number
  ): ISourceEvent | void;

  abstract release(note: number, when: number): void;
}
