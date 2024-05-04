import { AudioCanvas } from "./AudioCanvas";
import { AudioClock } from "./AudioClock";

const DEFAULT_NOTE_HEIGHT = 20;
const note_names = [
  "A",
  "A#",
  "B",
  "C",
  "C#",
  "D",
  "D#",
  "E",
  "F",
  "F#",
  "G",
  "G#",
];
const keyColors = [
  "white",
  "black",
  "white",
  "white",
  "black",
  "white",
  "black",
  "white",
  "white",
  "black",
  "white",
  "black",
];

let audioContext: AudioContext | null = null;

function getAudioContext() {
  if (!audioContext) {
    const AudioContext =
      window.AudioContext || (window as any).webkitAudioContext;
    audioContext = new AudioContext();
  }
  return audioContext;
}

function getNoteNameFromPitch(pitch: number): string {
  const note = note_names[pitch % note_names.length];
  return `${note}${Math.floor(pitch / note_names.length)}`;
}

export class PianoRollDialog {
  domElement: HTMLDivElement;
  closeButton: HTMLButtonElement;
  saveButton: HTMLButtonElement;
  note: GridItem | null = null;

  constructor(public onsave: (note: GridItem) => void) {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll-dialog");

    this.closeButton = document.createElement("button");
    this.closeButton.textContent = "X";
    this.domElement.appendChild(this.closeButton);
    this.closeButton.addEventListener("click", () => {
      this.domElement.style.display = "none";
    });

    this.saveButton = document.createElement("button");
    this.saveButton.textContent = "Save";
    this.domElement.appendChild(this.saveButton);
    this.saveButton.addEventListener("click", () => {
      this.onsave(this.note!);
    });
  }

  show(note: GridItem, x: number, y: number) {
    console.log(note);
    this.note = note;
    this.domElement.style.display = "block";
    this.domElement.style.left = `${x}px`;
    this.domElement.style.top = `${y}px`;
  }
}

export class Composition {
  title: string;
  description: string;
  tempo: number;
  events: GridItem[] = [];

  constructor() {
    this.title = "Untitled";
    this.description = "No description";
    this.tempo = 120;
  }
}

export interface IGridItem {
  pitch: number;
  start: number;
  end: number;
  label: string;
}

export class GridItem implements IGridItem {
  pitch: number;
  start: number;
  end: number;
  label: string | "" = "";
  domElement: HTMLDivElement;
  noteLabel: HTMLDivElement;
  lyricLabel: HTMLDivElement;
  audio: AudioBuffer | null = null;

  constructor(
    private readonly grid: Grid,
    item: IGridItem
  ) {
    this.pitch = item.pitch;
    this.start = item.start;
    this.end = item.end;
    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll-note");
    this.domElement.style.height = `${grid.noteHeight}px`;

    this.noteLabel = document.createElement("div");
    this.noteLabel.classList.add("piano-roll-note-label");
    this.noteLabel.textContent = getNoteNameFromPitch(this.pitch);
    this.domElement.appendChild(this.noteLabel);

    this.lyricLabel = document.createElement("div");
    this.lyricLabel.classList.add("piano-roll-note-lyric");
    this.domElement.appendChild(this.lyricLabel);

    if (item.label) this.lyricLabel.textContent = item.label;
    this.update();

    grid.domElement.appendChild(this.domElement);
  }

  update() {
    this.noteLabel.textContent = getNoteNameFromPitch(this.pitch);
    this.lyricLabel.textContent = this.label;
    this.domElement.style.top = `${(87 - this.pitch) * this.grid.noteHeight}px`;
    this.domElement.style.left = `${this.start * this.grid.beatWidth}%`;
    this.domElement.style.width = `${(this.end - this.start) * 100}%`;
  }
}

class LyricEditorDialog extends PianoRollDialog {
  noteText: HTMLInputElement;
  audioCanvas: AudioCanvas;
  constructor(onsave: (note: GridItem) => void) {
    super(onsave);
    this.domElement.classList.add("piano-roll-note-editor");
    this.domElement.style.display = "none";

    this.noteText = document.createElement("input");
    this.noteText.type = "text";
    this.domElement.appendChild(this.noteText);
    this.noteText.setAttribute("placeholder", "Note lyric");

    this.audioCanvas = new AudioCanvas(getAudioContext());
    this.domElement.appendChild(this.audioCanvas.domElement);

    this.saveButton.addEventListener("click", () => {
      this.audioCanvas.domElement.style.display = "block";
      this.audioCanvas.generateAudio(this.noteText.value, true).then((buffer) => {
        this.note!.audio = buffer;                
      });
      this.onsave(this.note!);
    });
  }

  show(note: GridItem, x: number, y: number) {
    super.show(note, x, y);
    this.noteText.value = this.note?.label || "";
    if (this.note?.audio) {
      this.audioCanvas.loadBuffer(this.note.audio);
      this.audioCanvas.domElement.style.display = "block";
    } else {
      this.audioCanvas.domElement.style.display = "none";
    }
  }
}

export class Grid {
  domElement: HTMLDivElement;
  gridElement: HTMLDivElement;
  notes: GridItem[] = [];
  currentNote: GridItem | null = null;
  noteHeight = DEFAULT_NOTE_HEIGHT;
  beatWidth = 100;
  noteEditor: LyricEditorDialog;

  constructor() {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll-grid-container");

    this.noteEditor = new LyricEditorDialog((note) => {
      note.label =
        this.noteEditor.domElement.querySelector("input")?.value || "";
      note.update();
    });

    document.body.appendChild(this.noteEditor.domElement);

    this.gridElement = document.createElement("div");
    this.gridElement.classList.add("piano-roll-grid");
    for (let i = 0; i < 88; i++) {
      const row = document.createElement("div");
      row.classList.add("piano-roll-grid-row");
      row.style.height = `${this.noteHeight}px`;
      this.gridElement.appendChild(row);
    }
    this.domElement.appendChild(this.gridElement);

    this.gridElement.addEventListener("pointerdown", (event) => {
      event.preventDefault();      
      if (this.noteEditor.domElement.style.display === "block") {
        this.noteEditor.domElement.style.display = "none";
        return;
      }
      if (event.button !== 0) return;
      if (event.target === this.gridElement) {
        this.gridElement.classList.add("dragging");
        const pitch = 87 - Math.floor(event.layerY / this.noteHeight);

        this.currentNote = new GridItem(this, {
          pitch: pitch,
          start: event.layerX / this.gridElement.clientWidth,
          end: event.layerX / this.gridElement.clientWidth,
          label: "",
        });
        this.add(this.currentNote);
      }
    });

    this.gridElement.addEventListener("pointerup", (event) => {
      this.gridElement.classList.remove("dragging");
      this.currentNote = null;
    });

    this.gridElement.addEventListener("pointerleave", (event) => {
      this.gridElement.classList.remove("dragging");
      if (this.currentNote) {
        this.remove(this.currentNote);
        this.currentNote = null;
      }
    });

    this.gridElement.addEventListener("contextmenu", (event) => {
      event.preventDefault();
    });

    this.gridElement.addEventListener("wheel", (event) => {
      if (this.currentNote) {
        this.currentNote.pitch += event.deltaY / 100;
        this.currentNote.domElement.style.top = `${
          this.currentNote.pitch * 10
        }px`;
      }
    });

    const noteEditor = this.noteEditor;

    this.gridElement.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      if (event.target instanceof HTMLDivElement) {
        this.gridElement.classList.add("dragging");
        const note = this.notes.find((n) => n.domElement === event.target);

        if (event.ctrlKey || event.button === 1) {
          if (note)
            noteEditor.show(note, event.clientX + 20, event.clientY - 5);
          else noteEditor.domElement.style.display = "none";
        }

        if (note) {
          console.log(event);
          this.currentNote = note;

          if (event.button === 2) {
            this.remove(note);
            this.currentNote = null;
          }
        }
      }
    });

    this.gridElement.addEventListener("pointermove", (event) => {
      // drag note longer
      if (this.currentNote && event.buttons === 1) {
        this.currentNote.end = event.layerX / this.gridElement.clientWidth;
        this.currentNote.domElement.style.width = `${
          (this.currentNote.end - this.currentNote.start) * 100
        }%`;
      }
    });

    this.gridElement.addEventListener("pointerup", (event) => {
      this.gridElement.classList.remove("dragging");
      if (this.currentNote) {
        this.currentNote.domElement.style.pointerEvents = "auto";
      }
      this.currentNote = null;
    });

    this.gridElement.addEventListener("pointerleave", (event) => {
      this.gridElement.classList.remove("dragging");
      if (this.currentNote) {
        this.currentNote.domElement.style.pointerEvents = "auto";
      }
      this.currentNote = null;
    });

    this.update();
  }

  update() {
    // TODO: update background grid

    this.notes.forEach((note) => note.update());
  }

  add(note: GridItem) {
    this.notes.push(note);
    this.gridElement.appendChild(note.domElement);
  }
  remove(note: GridItem) {
    this.notes = this.notes.filter((n) => n !== note);
    this.gridElement.removeChild(note.domElement);
  }
  load(composition: object) {
    const comp = new Composition();
    if ("title" in composition) comp.title = composition["title"] as string;
    if ("description" in composition)
      comp.description = composition["description"] as string;
    if ("tempo" in composition) comp.tempo = composition["tempo"] as number;
    if ("events" in composition)
      comp.events = (composition["events"] as IGridItem[]).map(
        (e: IGridItem) => new GridItem(this, e)
      );
  }
  download(): Composition {
    const comp = new Composition();
    comp.events = this.notes;

    const data = JSON.stringify(comp);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "composition.json";
    a.click();
    URL.revokeObjectURL(url);

    return comp;
  }
}

export class PianoRoll {
  domElement: HTMLDivElement;
  grid: Grid;
  sideKeyboard: SideKeyboard;
  cursor: HTMLDivElement;
  cursorUpdateInterval: any;

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
    });

    clock.on("stop", () => {
      this.cursor.style.display = "none";
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
        keyColors[i % 12] === "white" ? "#fff" : "#000";
      key.style.color = keyColors[i % 12] === "white" ? "#000" : "#fff";
      key.textContent =
        note_names[i % 12] + ((i / note_names.length) | 0).toString();
      this.domElement.appendChild(key);
    }
  }
}
