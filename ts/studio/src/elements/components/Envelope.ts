import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { IEnvelope } from "../../schema";
import { Slider } from "./Slider";

export class Envelope extends BaseElement<"update"> implements IEnvelope {
  readonly attack: Slider;
  readonly hold: Slider;
  readonly decay: Slider;
  readonly sustain: Slider;
  readonly release: Slider;

  constructor(
    attack = 0.003,
    hold = 0.0,
    decay = 2.0,
    sustain = 0.01,
    release = 0.05
  ) {
    super("div", "envelope");

    this.attack = new Slider("attack", "A", 0, 1, 0.01, attack);
    this.hold = new Slider("hold", "H", 0, 1, 0.01, hold);
    this.decay = new Slider("decay", "D", 0, 1, 0.01, decay);
    this.sustain = new Slider("sustain", "S", 0, 1, 0.01, sustain);
    this.release = new Slider("release", "R", 0, 1, 0.01, release);

    this.domElement.appendChild(this.attack.domElement);
    this.domElement.appendChild(this.decay.domElement);
    this.domElement.appendChild(this.sustain.domElement);
    this.domElement.appendChild(this.release.domElement);
  }

  trigger(param: AudioParam, when: number) {
    console.log("trigger @ " + when);
    const value = param.value;
    param.cancelScheduledValues(when);
    param.setValueAtTime(value, when);
    param.exponentialRampToValueAtTime(0.015, when);
    //param.setValueAtTime(0.015, when);
    param.linearRampToValueAtTime(1, when + this.attack.value);
    //param.setValueAtTime(1, when + this.attack.value);
    param.exponentialRampToValueAtTime(
      this.sustain.value,
      when + this.attack.value + this.decay.value
    );
  }

  triggerRelease(param: AudioParam, when: number) {
    console.log("triggerRelease @ " + when);
    param.setValueAtTime(param.value, when);
    param.setTargetAtTime(0.0001, when + 0.01, this.release.value);
    param.cancelScheduledValues(when + 0.01 + this.release.value);
  }
}
