import {
  IInstrumentWindowOptions,
  InstrumentWindow,
} from "../../abstracts/InstrumentWindow";
import type { ProjectUI } from "../../elements/ProjectUI";
import type { Multisampler } from "./Multisampler";
import { Sampler } from "./Sampler";

export interface IMultisamplerWindowOptions extends IInstrumentWindowOptions {
  instrument: Multisampler;
}

export class MultisamplerWindow extends InstrumentWindow {
  private readonly _slotsContainer: HTMLDivElement;

  constructor(
    readonly ui: ProjectUI,
    readonly instrument: Multisampler
  ) {
    super(ui, instrument);

    // window.addEventListener("keydown", (event) => this.onKeyDown(event));

    const container = document.createElement("div");
    container.classList.add("sampler-slots-container");
    this._slotsContainer = container;

    //keypad numbers 0-9
    // then . / * + - Enter
    const defaultSlots = [
      { keyBinding: 96, name: "0" },
      { keyBinding: 97, name: "1" },
      { keyBinding: 98, name: "2" },
      { keyBinding: 99, name: "3" },
      { keyBinding: 100, name: "4" },
      { keyBinding: 101, name: "5" },
      { keyBinding: 102, name: "6", cutByGroups: [3, 6] },
      { keyBinding: 103, name: "7" },
      { keyBinding: 104, name: "8" },
      { keyBinding: 105, name: "9" },
      { keyBinding: 111, name: "/", cutByGroups: [0, 10] },
      { keyBinding: 106, name: "*" },
      { keyBinding: 109, name: "-" },
      { keyBinding: 107, name: "+" },
      { keyBinding: 13, name: "Enter" },
    ];

    for (let i = 0; i < defaultSlots.length; i++) {
      const slot = new Sampler(this.ui.project, { name: defaultSlots[i].name });
      instrument.samplers.push(slot);
      slot.loadUrl(`./data/samples/kit1/${i}.wav`);
    }

    ui.project.audioClock.on("stop", () => {
      for (const slot of instrument.samplers) {
        slot.stop();
      }
    });

    this.content.appendChild(container);
  }

  // onKeyDown(event: KeyboardEvent) {
  //   //console.log(event.key, event.keyCode);
  //   const slot = this.options.instrument.samplers.find(
  //     (slot) => slot.keyBinding === event.keyCode
  //   );
  //   if (slot) {
  //     event.preventDefault();
  //     const note = this.instrument.samplers.indexOf(slot);
  //     this.instrument.trigger(note, null);
  //   }
  // }
}
