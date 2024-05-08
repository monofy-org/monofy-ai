import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { AudioClock } from "../components/AudioClock";
import { SamplerSlot } from "../components/SamplerSlot";

export class SamplerWindow extends DraggableWindow<"update"> {
  slots: SamplerSlot[] = [];
  private _slotsContainer: HTMLDivElement;

  constructor(readonly audioClock: AudioClock) {
    const container = document.createElement("div");
    container.classList.add("sampler-slots-container");

    super("Sampler", true, container);
    this._slotsContainer = container;

    this.setSize(400, 400);

    //keypad numbers 0-9
    // then . / * + - Enter
    const defaultSlots = [
      { keyBinding: 96, name: "0" },
      { keyBinding: 97, name: "1" },
      { keyBinding: 98, name: "2" },
      { keyBinding: 99, name: "3" },
      { keyBinding: 100, name: "4" },
      { keyBinding: 101, name: "5" },
      { keyBinding: 102, name: "6" },
      { keyBinding: 103, name: "7" },
      { keyBinding: 104, name: "8" },
      { keyBinding: 105, name: "9" },
      { keyBinding: 111, name: "/" },
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
        i
      );
      this.slots.push(slot);
      this._slotsContainer.appendChild(slot.domElement);
      slot.loadSample("Sample", `./data/samples/kit1/${i}.wav`);
    }

    window.addEventListener("keydown", (event) => this.onKeyDown(event));
  }

  onKeyDown(event: KeyboardEvent) {
    //console.log(event.key, event.keyCode);
    const slot = this.slots.find((slot) => slot.keyBinding === event.keyCode);
    if (slot) {
      event.preventDefault();
      slot.trigger();
    }
  }
}
