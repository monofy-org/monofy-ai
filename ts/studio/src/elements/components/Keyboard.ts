import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { BOTTOM_ROW, TOP_ROW } from "../../constants";

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
}
