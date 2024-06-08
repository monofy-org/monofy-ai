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
  private _output: GainNode;  
  private _held: ISynthesizerEvent[] = [];
  public transpose = 24;

  get output() {
    return this._output;
  }

  constructor(audioClock: AudioClock) {
    super(audioClock);    
    this._output = audioClock.audioContext.createGain();    
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

      const index = this._held.findIndex((e) => e.note == event.note);
      if (index >= 0) {
        this._held.splice(index, 1);
      }

      this.voices.forEach((voice) => {
        voice.release(event.note);
      });
    } else {
      console.error(this.name + ": Invalid event passed to trigger()", event);
    }
  }
}
