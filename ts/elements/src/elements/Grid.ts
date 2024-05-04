import { DEFAULT_NOTE_HEIGHT, NOTE_NAMES } from "../constants/audioConstants";
import { Composition } from "./Composition";
import { LyricEditorDialog } from "./audioDialogs";

function getNoteNameFromPitch(pitch: number): string {
  const note = NOTE_NAMES[pitch % NOTE_NAMES.length];
  return `${note}${Math.floor(pitch / NOTE_NAMES.length)}`;
}

const NOTE_HANDLE_WIDTH = 6;

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
    this.domElement.style.left = `${this.start * this.grid.beatWidth}px`;
    this.domElement.style.width = `${(this.end - this.start) * 100}px`;
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
  dragMode: string | null = null;

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
          start: event.layerX / this.beatWidth,
          end: (event.layerX + 0.125) / this.beatWidth,
          label: "",
        });
        this.add(this.currentNote);
        this.dragMode = "end";
      }
    });

    this.gridElement.addEventListener("pointerup", () => {
      this.gridElement.classList.remove("dragging");
      this.currentNote = null;
    });

    this.gridElement.addEventListener("pointerleave", () => {
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
          } else {
            this.currentNote.domElement.style.zIndex = "100";
            if (event.layerX < NOTE_HANDLE_WIDTH) {
              this.dragMode = "start";
            } else if (
              event.layerX >
              this.currentNote.domElement.offsetWidth - NOTE_HANDLE_WIDTH
            ) {
              this.dragMode = "end";
            } else {
              this.dragMode = "move";
            }
          }
        }
      }
    });

    this.gridElement.addEventListener("pointermove", (event) => {
      // drag note longer
      console.log(this.dragMode);
      if (this.currentNote && event.buttons === 1) {
        if (this.dragMode === "start") {
          this.currentNote.start = event.layerX / this.beatWidth;
          this.currentNote.domElement.style.left = `${
            this.currentNote.start * this.beatWidth
          }px`;
          this.currentNote.domElement.style.width = `${
            (this.currentNote.end - this.currentNote.start) * this.beatWidth
          }px`;
        } else if (this.dragMode === "end") {
          this.currentNote.end = event.layerX / this.beatWidth;
          this.currentNote.domElement.style.width = `${
            (this.currentNote.end - this.currentNote.start) * this.beatWidth
          }px`;
        } else if (this.dragMode === "move") {
          const delta = event.movementX / this.beatWidth;
          this.currentNote.start += delta;
          this.currentNote.end += delta;
          this.currentNote.domElement.style.left = `${
            this.currentNote.start * this.beatWidth
          }px`;
        }
      }
    });

    this.gridElement.addEventListener("pointerup", () => {
      this.gridElement.classList.remove("dragging");
      if (this.currentNote) {
        this.currentNote.domElement.style.pointerEvents = "auto";
      }
      this.currentNote = null;
    });

    this.gridElement.addEventListener("pointerleave", () => {
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
