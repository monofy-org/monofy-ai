import { SynthesizerVoice } from "./SynthesizerVoice";
import { IKeyboardEvent } from "../elements/components/Keyboard";
import { Instrument } from "./Instrument";
import { Project } from "../elements/Project";

interface ISynthesizerEvent extends IKeyboardEvent {
  voice: SynthesizerVoice;
}

export abstract class Synthesizer
  extends Instrument  
{
  readonly voices: SynthesizerVoice[] = [];
  private _nextVoice = 0;
  private _held: ISynthesizerEvent[] = [];

  constructor(
    project: Project,
    voiceType: typeof SynthesizerVoice,
    mixerChannel = 0,
    maxPoly = 16
  ) {
    super(project, mixerChannel);

    for (let i = 0; i < maxPoly; i++) {
      const voice: SynthesizerVoice = new (voiceType as any)(
        project.audioClock
      );
      this.voices.push(voice);
      voice.output.connect(this.output);
    }
  }

  trigger(note: number, when: number, velocity = 1) {
    console.log(this.name + " triggered voice " + this._nextVoice);

    if (this._held.find((e) => e.note == note)) {
      console.warn(this.name + " already holding note " + note);
      return;
    }

    this._held.push({
      note,
      type: "press",
      voice: this.voices[this._nextVoice],
      channel: 0,
      velocity: velocity || 1,
    });

    this.voices[this._nextVoice].trigger(note, when, velocity);

    this._nextVoice++;
    if (this._nextVoice >= this.voices.length) {
      this._nextVoice = 0;
    }
  }

  release(note: number, when: number) {
    console.log(this.name + " released voice " + note);

    const index = this._held.findIndex((e) => e.note == note);
    if (index >= 0) {
      this._held[index].voice.release(note, when);
      this._held.splice(index, 1);
    } else {
      console.warn(this.name + " not holding note " + note);
    }
  }
}
