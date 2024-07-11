import { InstrumentWindow } from "../../abstracts/InstrumentWindow";
import { Synthesizer } from "../../abstracts/Synthesizer";
import { SynthesizerVoice } from "../../abstracts/SynthesizerVoice";
import type { Project } from "../../elements/Project";
import type { AudioClock } from "../../elements/components/AudioClock";
import { Envelope } from "../../elements/components/Envelope";
import type { ControllerGroup } from "../../schema";

function noteToFrequency(note: number) {
  return 440 * Math.pow(2, (note - 69) / 12);
}

export class FMPianoVoice extends SynthesizerVoice {
  readonly name = "FM Piano Voice";
  readonly id = "fm_piano_voice";

  private _carrier1?: OscillatorNode;
  private _modulator1?: OscillatorNode;
  private readonly _gain1: GainNode;
  readonly modulatorGain1: GainNode;
  private readonly _filter: BiquadFilterNode;
  readonly filterEnvelope: Envelope;
  readonly gainEnvelope: Envelope;

  public fmRatio = 1;

  constructor(readonly audioClock: AudioClock) {
    super(audioClock);

    const audioContext = audioClock.audioContext;

    this.modulatorGain1 = audioContext.createGain();
    this.modulatorGain1.gain.value = 100;

    this._gain1 = audioContext.createGain();
    this._gain1.gain.value = 0;

    const highpassFilter = audioContext.createBiquadFilter();
    highpassFilter.type = "highpass";
    highpassFilter.frequency.value = 20;
    this._gain1.connect(highpassFilter);
    highpassFilter.connect(this.output);

    this.output.gain.value = 0.5;

    this._filter = audioContext.createBiquadFilter();
    this._filter.type = "lowpass";
    this._filter.frequency.value = 440;
    this._filter.Q.value = 1;
    this._filter.connect(this._gain1);

    this.filterEnvelope = new Envelope();
    this.gainEnvelope = new Envelope({
      attack: 0,
      hold: 0,
      decay: 2,
      sustain: 0,
      release: 0.3,
    });
  }

  trigger(note: number, when: number, velocity: number) {
    console.log("FMPianoVoice triggered", note, when, velocity);

    if (this._carrier1?.context.state === "running") {
      const g = this._gain1.gain.value;
      this._gain1.gain.cancelScheduledValues(when);
      this._gain1.gain.setValueAtTime(g, when);
      this._gain1.gain.exponentialRampToValueAtTime(1, when + 0.01);

      try {
        this._carrier1.stop(when + 0.02);
        this._modulator1?.stop(when + 0.02);
      } catch (e) {
        console.error("Error stopping carrier/modulator", e);
      }
    }

    const freq = noteToFrequency(note);

    this._carrier1 = this.audioClock.audioContext.createOscillator();
    this._carrier1.frequency.value = freq;

    this._modulator1 = this.audioClock.audioContext.createOscillator();
    this._modulator1.frequency.value = freq * this.fmRatio;

    this._carrier1.connect(this._gain1);
    this._modulator1.connect(this.modulatorGain1);
    this.modulatorGain1.connect(this._carrier1.frequency);

    this._carrier1.type = "sine";

    this.output.gain.cancelScheduledValues(when);
    this.output.gain.setValueAtTime(velocity, when);
    this._carrier1.start(when);
    this._modulator1.start(when);

    const time = when || this.audioClock.currentTime;

    this.gainEnvelope.trigger(this._gain1.gain, time);

    //this.filterEnvelope.trigger(this.filter.frequency, time);

    return undefined;
  }

  release(note: number, when: number) {
    const time = when || this.audioClock.currentTime;
    //this.filterEnvelope.triggerRelease(this._filter.frequency, time);
    this.gainEnvelope.triggerRelease(this.output.gain, time);
  }
}

export class FMPiano extends Synthesizer {
  readonly Window = InstrumentWindow;

  readonly name = "FM Piano";
  readonly id = "fm_piano";
  readonly version = "0.0.1";
  readonly description = "A simple FM piano synthesizer";
  readonly author = "Johnny Street";
  readonly controllerGroups: ControllerGroup[] = [];

  constructor(project: Project, mixerChannel = 0) {
    super(project, FMPianoVoice, mixerChannel);
  }
}
