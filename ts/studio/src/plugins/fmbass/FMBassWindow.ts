import {
  InstrumentWindow,
} from "../../abstracts/InstrumentWindow";
import type { ProjectUI } from "../../elements/ProjectUI";
import { Knob } from "../../elements/components/Knob";
import { FMBass } from "./FMBass";


export class FMBassWindow extends InstrumentWindow {

  instrument: FMBass;

  constructor(
    readonly ui: ProjectUI,
    readonly fmbass: FMBass
  ) {
    super(ui, fmbass);

    this.instrument = fmbass;

    const fmGainKnob = new Knob({
      controlType: "knob",
      name: "FM Gain",
      label: "FM Gain",
      min: 0,
      max: 1000,
      step: 1,
      default: 100,
      value: 100,
    });

    this.content.appendChild(fmGainKnob.domElement);
    fmGainKnob.on("change", () => {
      console.log("FM Gain", fmGainKnob.value);
      this.instrument.modulatorGain.gain.value = fmGainKnob.value;
    });

    const fmRatioKnob = new Knob({
      controlType: "knob",
      name: "FM Ratio",
      label: "FM Ratio",
      min: 0,
      max: 12,
      step: 1,
      default: this.instrument.fmRatio,
      value: this.instrument.fmRatio,
    });
    fmRatioKnob.on("change", () => {
      console.log("FM Ratio", fmRatioKnob.value);
      this.instrument.fmRatio = fmRatioKnob.value;
    });
    this.content.appendChild(fmRatioKnob.domElement);
  }
}
