import { Instrument } from "../../abstracts/Instrument";
import { SynthesizerVoice } from "../../abstracts/SynthesizerVoice";
import type { Project } from "../../elements/Project";
import type { AudioClock } from "../../elements/components/AudioClock";
import { Envelope } from "../../elements/components/Envelope";
import type { ControllerGroup } from "../../schema";
import { FMBassWindow } from "./FMBassWindow";

function noteToFrequency(note: number) {
  return 440 * Math.pow(2, (note - 69) / 12);
}

export class FMBassVoice extends SynthesizerVoice {

  readonly name = "FM Bass Voice";
  readonly id = "fm_bass_voice";
  
  constructor(readonly audioClock: AudioClock) {
    super(audioClock);
  }

  trigger(note: number, when: number, velocity?: number) {
    console.log("FMBassVoice triggered", note, when, velocity);
  }

  release(note: number) {
    console.log("FMBassVoice released", note);
  }
}

export class FMBass extends Instrument {
  readonly name = "FM Bass";
  readonly id = "fm_bass";
  readonly version = "0.0.1";
  readonly description = "A simple FM bass synthesizer";
  readonly author = "Johnny Street";
  readonly controllerGroups: ControllerGroup[] = [];
  private _carrier?: OscillatorNode;
  private _modulator?: OscillatorNode;
  private readonly _gain: GainNode;
  readonly modulatorGain: GainNode;
  private readonly _filter: BiquadFilterNode;
  readonly filterEnvelope: Envelope;
  readonly gainEnvelope: Envelope;

  readonly Window = FMBassWindow;

  private _heldNote: number | undefined;

  public fmRatio = 1;
  private _window?: FMBassWindow;

  constructor(project: Project, mixerChannel = 0) {
    super(project, mixerChannel);

    const audioContext = project.audioClock.audioContext;

    this.modulatorGain = audioContext.createGain();
    this.modulatorGain.gain.value = 100;

    this._gain = audioContext.createGain();
    this._gain.gain.value = 0;

    const highpassFilter = audioContext.createBiquadFilter();
    highpassFilter.type = "highpass";
    highpassFilter.frequency.value = 20;
    this._gain.connect(highpassFilter);
    highpassFilter.connect(this.output);

    this.output.gain.value = 0.5;

    this._filter = audioContext.createBiquadFilter();
    this._filter.type = "lowpass";
    this._filter.frequency.value = 440;
    this._filter.Q.value = 1;
    this._filter.connect(this._gain);

    this.filterEnvelope = new Envelope();
    this.gainEnvelope = new Envelope();
  }

  trigger(note: number, when: number, velocity = 1) {
    console.log("FM Bass triggered", when, "velocity = ", velocity);

    this._heldNote = note;

    note += this.transpose;

    if (this._carrier?.context.state === "running") {
      const g = this._gain.gain.value;
      this._gain.gain.cancelScheduledValues(when);
      this._gain.gain.setValueAtTime(g, when);
      this._gain.gain.exponentialRampToValueAtTime(velocity * 0.1, when + 0.01);

      try {
        this._carrier.stop(when + 0.02);
        this._modulator?.stop(when + 0.02);
      } catch (e) {
        console.error("Error stopping carrier/modulator", e);
      }
    }

    const freq = noteToFrequency(note);

    this._carrier = this.project.audioClock.audioContext.createOscillator();
    this._carrier.frequency.value = freq;

    this._modulator = this.project.audioClock.audioContext.createOscillator();
    this._modulator.frequency.value = freq * this.fmRatio;

    this._carrier.connect(this._gain);
    this._modulator.connect(this.modulatorGain);
    this.modulatorGain.connect(this._carrier.frequency);

    this._carrier.type = "sine";

    this._carrier.start(when);
    this._modulator.start(when);

    const time = when || this.project.audioClock.currentTime;

    this.gainEnvelope.trigger(this._gain.gain, time);

    //this.filterEnvelope.trigger(this.filter.frequency, time);

    return undefined;
  }

  release(note: number, when: number) {
    if (this._heldNote !== note) {
      return;
    }
    console.log("FM Bass released", note);
    const time = when || this.project.audioClock.currentTime;
    this.filterEnvelope.triggerRelease(this._filter.frequency, time);
    this.gainEnvelope.triggerRelease(this._gain.gain, time);
  }
}
