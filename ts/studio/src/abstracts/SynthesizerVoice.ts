import { AudioClock } from "../elements/components/AudioClock";
import { ISynthesizerVoiceSettings } from "../schema";

export abstract class SynthesizerVoice implements ISynthesizerVoiceSettings {
  _settings: ISynthesizerVoiceSettings | null = null;

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
    this._gain.gain.value = settings.gain;
  }

  trigger(note: number, velocity: number, time = 0) {
    if (!this.settings) throw new Error("Voice not initialized");

    this._note = note;
    this._frequency = 440 * Math.pow(2, (note - 69) / 12);
    this._gain.gain.value = this.settings.gain * velocity;

    // TODO
  }

  release(time = 0) {
    if (!this.settings) throw new Error("Voice not initialized");

    this._note = null;
    this._frequency = null;

    // TODO
  }
}
