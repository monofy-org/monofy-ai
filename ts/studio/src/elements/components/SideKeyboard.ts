import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import {
  DEFAULT_NOTE_HEIGHT,
  KEY_COLORS,
  NOTE_NAMES,
} from "../../constants";
import { IKeyboardEvent } from "./Keyboard";

export class SideKeyboard extends BaseElement<"update"> {  
  keys: HTMLDivElement[] = [];
  noteHeight = DEFAULT_NOTE_HEIGHT;

  constructor() {

    super("div", "piano-roll-keyboard");

    this.redraw();

    this.domElement.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      const key = e.target as HTMLDivElement;

      if (key.classList.contains("piano-roll-keyboard-key")) {
        const note_index = this.keys.indexOf(key);
        console.log("key", key.textContent, note_index);        
        key.classList.add("active");
        this.emit("update", { type: "press", note: note_index } as IKeyboardEvent);
        const pointerup = () => {
          window.removeEventListener("pointerup", pointerup);
          key.classList.remove("active");
          this.emit("update", { type: "release", note: note_index } as IKeyboardEvent);          
        };

        window.addEventListener("pointerup", () => pointerup());
      }
    });
  }

  redraw() {
    this.domElement.innerHTML = "";

    for (let i = 87; i >= 0; i--) {
      const key = document.createElement("div");
      key.classList.add("piano-roll-keyboard-key");
      key.style.height = `${this.noteHeight}px`;
      // key.style.bottom = `${i * this.noteHeight}px`;
      key.style.backgroundColor =
        KEY_COLORS[i % 12] === "white" ? "#fff" : "#000";
      key.style.color = KEY_COLORS[i % 12] === "white" ? "#000" : "#fff";
      key.textContent = NOTE_NAMES[i % 12] + ((i / 12) | 0).toString();
      this.domElement.appendChild(key);
      this.keys.push(key);
    }

    this.keys.reverse();
  }
}
