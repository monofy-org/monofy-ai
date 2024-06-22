import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { PopupNotification } from "../../../../elements/src/elements/PopupNotification";
import { BOTTOM_ROW, TOP_ROW } from "../../constants";

let midiAccess: MIDIAccess | null = null;

export interface IKeyboardEvent {
  type: "press" | "release";
  note: number;
}

function keyToNote(keyCode: number) {
  let note = -1;

  if (BOTTOM_ROW.includes(keyCode)) {
    note = BOTTOM_ROW.indexOf(keyCode);
  } else if (TOP_ROW.includes(keyCode)) {
    note = TOP_ROW.indexOf(keyCode) + 12;
  }

  return note;
}

export class Keyboard extends BaseElement<"update"> {
  constructor() {
    super("div", "keyboard");

    if (!midiAccess) {
      navigator
        .requestMIDIAccess()
        .then((midi) => {
          console.log("MIDI Access", midi);
          midiAccess = midi;
          midi.inputs.forEach((entry) => {
            entry.onmidimessage = (e) => this.handleMIDI(e);
          });
        })
        .catch((e) => {
          console.error("MIDI Access error", e);
          new PopupNotification("MIDI Access error", { body: e.message });
        });
    }

    const pressedKeys: number[] = [];

    window.addEventListener("keydown", (event) => {
      //console.log(event.key, event.keyCode);

      if (event.ctrlKey || event.altKey || event.shiftKey) return;

      const target = event.target as HTMLElement;
      if (target.tagName == "INPUT") return;

      const note = keyToNote(event.keyCode);
      if (note === -1 || pressedKeys.includes(note)) return;

      pressedKeys.push(note);
      this.emit("update", { type: "press", note });
    });

    window.addEventListener("keyup", (event) => {
      const note = keyToNote(event.keyCode);
      if (note === -1 || !pressedKeys.includes(note)) return;

      pressedKeys.splice(pressedKeys.indexOf(note), 1);
      this.emit("update", { type: "release", note });
    });
  }

  private handleMIDI(event: MIDIMessageEvent) {
    console.log("MIDI", event.data?.length, "bytes", event.data);
    if (event.data === null || event.data.length < 3) return;

    // event.data is a uint8array

    const statusByte = event.data[0];
    const eventType = statusByte & 0xf0;
    const channel = statusByte & 0x0f;

    const dataByte1 = event.data[1];
    const dataByte2 = event.data[2];

    if (eventType === 0x90 && dataByte2 !== 0) {
      // Note On event
      const note = dataByte1 - 36;
      const velocity = dataByte2;
      this.emit("update", { type: "press", note, velocity });
    } else if (eventType === 0x80 || (eventType === 0x90 && dataByte2 === 0)) {
      // Note Off event
      const note = dataByte1 - 36;
      this.emit("update", { type: "release", note });
    } else if (eventType === 0xb0) {
      // Control Change event
      const control = dataByte1;
      const value = dataByte2;
      // Process the control change event
      console.log("Control Change", control, value);
    }
  }
}
