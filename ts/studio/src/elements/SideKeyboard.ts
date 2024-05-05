import {
  DEFAULT_NOTE_HEIGHT,
  KEY_COLORS,
  NOTE_NAMES,
} from "../../../elements/src/constants/audioConstants";

export class SideKeyboard {
  domElement: HTMLDivElement;
  keys: HTMLDivElement[] = [];
  noteHeight = DEFAULT_NOTE_HEIGHT;

  constructor() {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll-keyboard");

    this.redraw();
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
    }
  }
}
