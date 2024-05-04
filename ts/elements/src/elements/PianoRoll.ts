import {
  DEFAULT_NOTE_HEIGHT,
  KEY_COLORS,
  NOTE_NAMES,
} from "../constants/audioConstants";
import { getAudioContext } from "../managers/AudioManager";
import { AudioClock } from "./AudioClock";
import { Grid } from "./Grid";


export class PianoRoll {
  domElement: HTMLDivElement;
  grid: Grid;
  sideKeyboard: SideKeyboard;
  cursor: HTMLDivElement;
  cursorUpdateInterval: any;
  scheduledSources: AudioBufferSourceNode[] = [];

  constructor(private readonly clock: AudioClock) {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll");

    this.sideKeyboard = new SideKeyboard();
    this.domElement.appendChild(this.sideKeyboard.domElement);

    this.grid = new Grid();
    this.domElement.appendChild(this.grid.domElement);

    this.cursor = document.createElement("div");
    this.cursor.classList.add("piano-roll-cursor");
    this.grid.domElement.appendChild(this.cursor);

    clock.on("start", () => {
      if (this.cursorUpdateInterval) {
        clearInterval(this.cursorUpdateInterval);
      }
      this.cursor.style.transform = "translateX(0)";
      this.cursor.style.display = "block";
      this.cursor.parentElement?.appendChild(this.cursor);
      requestAnimationFrame(this.updateCursor.bind(this));
      this.scheduleAudioEvents();
    });

    clock.on("stop", () => {
      this.cursor.style.display = "none";
      this.scheduledSources.forEach((source) => source.stop());
      this.scheduledSources = [];
    });
  }

  updateCursor() {
    if (this.clock.isPlaying) {
      this.cursor.style.transform = `translateX(${
        this.clock.currentBeat * this.grid.beatWidth
      }px)`;
      requestAnimationFrame(this.updateCursor.bind(this));
    }
  }

  scheduleAudioEvents() {
    const ctx = getAudioContext();
    this.grid.notes.forEach((note) => {
      const bufferSource = ctx.createBufferSource();
      bufferSource.buffer = note.audio;
      bufferSource.connect(ctx.destination);
      bufferSource.start(ctx.currentTime + (note.start * 60) / this.clock.bpm);
      this.scheduledSources.push(bufferSource);
    });
  }
}

class SideKeyboard {
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
      key.textContent =
        NOTE_NAMES[i % 12] + ((i / 12) | 0).toString();
      this.domElement.appendChild(key);
    }
  }
}
