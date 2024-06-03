import { getAudioContext } from "../../../elements/src/managers/AudioManager";
import { SynthesizerVoice } from "./SynthesizerVoice";
import { AudioClock } from "../elements/components/AudioClock";
import { IKeyboardEvent } from "../elements/components/Keyboard";
import { ISynthesizerSettings } from "../schema";
import { Instrument } from "./Instrument";
import { InstrumentWindow } from "./InstrumentWindow";

interface ISynthesizerEvent extends IKeyboardEvent {
  voice: SynthesizerVoice;
}

export abstract class Synthesizer<T extends SynthesizerVoice>
  extends Instrument
  implements ISynthesizerSettings
{
  abstract readonly window: InstrumentWindow;
  readonly voices: SynthesizerVoice[] = [];
  private _nextVoice = 0;
  private _gain: GainNode;
  private _ctx: AudioContext;
  private _held: ISynthesizerEvent[] = [];

  get gain() {
    return this._gain;
  }

  constructor(audioClock: AudioClock) {
    super(audioClock);
    this._ctx = getAudioContext();
    this._gain = this._ctx.createGain();
  }

  addVoice(voice: T) {
    this.voices.push(voice);
  }

  handleEvent(event: IKeyboardEvent) {
    if (event.type == "press") {
      console.log(this.name + " triggered voice " + this._nextVoice);

      this._held.push({ ...event, voice: this.voices[this._nextVoice] });

      this._nextVoice++;
      if (this._nextVoice >= this.voices.length) {
        this._nextVoice = 0;
      }
    } else if (event.type == "release") {
      console.log(this.name + " released voice " + event);
    } else {
      console.error(this.name + ": Invalid event passed to trigger()", event);
    }
  }
}
