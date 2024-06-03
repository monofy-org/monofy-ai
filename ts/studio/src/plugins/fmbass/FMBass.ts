import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { InstrumentWindow } from "../../abstracts/InstrumentWindow";
import { Synthesizer } from "../../abstracts/Synthesizer";
import { SynthesizerVoice } from "../../abstracts/SynthesizerVoice";
import { AudioClock } from "../../elements/components/AudioClock";
import { ISourceEvent } from "../../elements/components/SamplerSlot";
import { ControllerGroup } from "../../schema";
import { Plugins } from "../plugins";

export class FMBassVoice extends SynthesizerVoice {
  constructor(readonly audioClock: AudioClock) {
    super(audioClock);
  }

  trigger(note: number, when: number, velocity: number | undefined) {
    console.log("FMBassVoice triggered", note, when, velocity);
  }

  release(note: number) {
    console.log("FMBassVoice released", note);
  }
}

export class FMBass extends Synthesizer<FMBassVoice> {
  readonly name = "FM Bass";
  readonly id = "fm_bass";
  readonly version = "0.0.1";
  readonly description = "A simple FM bass synthesizer";
  readonly author = "Johnny Street";
  readonly controllerGroups: ControllerGroup[] = [];
  readonly window: InstrumentWindow;

  constructor(readonly audioClock: AudioClock) {
    super(audioClock);

    this.window = new DraggableWindow("FM Bass", true, this.domElement);
  }

  trigger(
    note: number,
    when: number,
    velocity: number | undefined
  ): ISourceEvent | undefined {
    console.log("FM Bass triggered", note, when, velocity);
    return undefined;
  }

  release(note: number) {
    console.log("FM Bass released", note);
  }
}

Plugins.register(FMBass);
