import { Instrument } from "../../abstracts/Instrument";
import { InstrumentWindow } from "../../abstracts/InstrumentWindow";
import { AudioClock } from "../../elements/components/AudioClock";
import { SamplerSlot } from "../../elements/components/SamplerSlot";
import { ControllerGroup } from "../../schema";
import { Plugins } from "../plugins";

export class Sampler extends Instrument {
  readonly name = "Sampler";
  readonly version = "0.0.1";
  readonly description = "A simple sampler";
  readonly author = "Johnny Street";
  readonly id = "sampler";
  readonly window: InstrumentWindow;

  private readonly _slotsContainer: HTMLDivElement;

  slots: SamplerSlot[] = [];
  controllerGroups: ControllerGroup[] = [];

  constructor(readonly audioClock: AudioClock) {
    super(audioClock);
    window.addEventListener("keydown", (event) => this.onKeyDown(event));

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

    for (let i = 0; i < 14; i++) {
      const slot = new SamplerSlot(
        this.audioClock,
        defaultSlots[i].name,
        defaultSlots[i].keyBinding,
        i,
        defaultSlots[i].cutByGroups
      );
      this.slots.push(slot);
      container.appendChild(slot.domElement);
      slot.loadSample("Sample", `./data/samples/kit1/${i}.wav`);
    }

    audioClock.on("stop", () => {
      for (const slot of this.slots) {
        slot.release(0);
      }
    });

    this.domElement.appendChild(container);

    this.window = new InstrumentWindow({
      title: "Sampler",
      persistent: true,
      content: this.domElement,
    });
  }

  trigger(note: number, channel: number | null, beat = 0) {
    console.log("Sampler trigger", note);
    const slot = this.slots[note];
    if (slot) {
      for (const each of this.slots) {
        if (each.cutByGroups.includes(slot.cutGroup)) {
          each.release(beat);
        }
      }
      return slot.trigger(beat);
    }
  }

  release(note: number, time = 0): void {
    if (this.slots[note]) {
      this.slots[note].release(time);
    }
  }

  onKeyDown(event: KeyboardEvent) {
    //console.log(event.key, event.keyCode);
    const slot = this.slots.find((slot) => slot.keyBinding === event.keyCode);
    if (slot) {
      event.preventDefault();
      const note = this.slots.indexOf(slot);
      this.trigger(note, null);
    }
  }
}

Plugins.register(Sampler);
