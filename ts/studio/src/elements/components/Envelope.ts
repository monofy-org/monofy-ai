import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { IEnvelope } from "../../schema";
import { Slider } from "./Slider";

export class Envelope extends BaseElement<"update"> implements IEnvelope {
  readonly attack: Slider;
  readonly hold: Slider;
  readonly decay: Slider;
  readonly sustain: Slider;
  readonly release: Slider;

  constructor() {
    super("div", "envelope");

    this.attack = new Slider("attack", "A", 0, 1, 0.01, 0.01);
    this.hold = new Slider("hold", "H", 0, 1, 0, 0);
    this.decay = new Slider("decay", "D", 0, 1, 0.01, 0.01);
    this.sustain = new Slider("sustain", "S", 0, 1, 0.01, 0.01);
    this.release = new Slider("release", "R", 0, 1, 0.01, 0.01);

    this.domElement.appendChild(this.attack.domElement);
    this.domElement.appendChild(this.decay.domElement);
    this.domElement.appendChild(this.sustain.domElement);
    this.domElement.appendChild(this.release.domElement);
  }

  trigger(param: AudioParam, time: number) {
    param.cancelScheduledValues(time);
    param.setValueAtTime(0, time);
    param.linearRampToValueAtTime(1, time + this.attack.value);
    param.linearRampToValueAtTime(
      this.sustain.value,
      time + this.attack.value + this.decay.value
    );
  }

  triggerRelease(param: AudioParam, time: number) {
    param.cancelScheduledValues(time);
    param.setValueAtTime(param.value, time);
    param.linearRampToValueAtTime(0, time + this.release.value);
  }
}
