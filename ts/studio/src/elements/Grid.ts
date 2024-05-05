import { DEFAULT_NOTE_HEIGHT, NOTE_NAMES } from "../../../elements/src/constants/audioConstants";
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
  quantize: number = 0.25;
  private dragOffset: number = 0;

  constructor() {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll-grid-container");

    this.gridElement = document.createElement("div");
    this.gridElement.classList.add("piano-roll-grid");

    const rowBackgroundCanvas = document.createElement("canvas");
    rowBackgroundCanvas.width = this.beatWidth * 4;
    rowBackgroundCanvas.height = this.noteHeight;
    console.log("debug", rowBackgroundCanvas.width, rowBackgroundCanvas.height);
    const ctx = rowBackgroundCanvas.getContext("2d");
    if (ctx) {
      ctx.strokeStyle = "#ddd";
      ctx.lineWidth = 1;
      for (let j = 1; j < 4; j++) {
        ctx.beginPath();
        ctx.moveTo(j * this.beatWidth, 0);
        ctx.lineTo(j * this.beatWidth, this.noteHeight);
        ctx.stroke();
      }
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#aaa";
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, this.noteHeight);
      ctx.stroke();
    }
    for (let i = 0; i < 88; i++) {
      const row = document.createElement("div");
      row.classList.add("piano-roll-grid-row");
      row.style.height = `${this.noteHeight}px`;
      this.gridElement.appendChild(row);
      row.style.backgroundImage = `url(${rowBackgroundCanvas.toDataURL()})`;
      row.style.backgroundRepeat = "repeat";
    }
    this.domElement.appendChild(this.gridElement);

    this.noteEditor = new LyricEditorDialog((note) => {
      const input = this.noteEditor.domElement.querySelector(
        ".lyric-editor-text"
      ) as HTMLInputElement;
      note.label = input?.value || "";
      note.update();
    });

    document.body.appendChild(this.noteEditor.domElement);

    this.gridElement.addEventListener("pointerdown", (event) => {
      console.log(event);
      const x = event.layerX;
      this.dragOffset = x;
      event.preventDefault();
      if (this.noteEditor.domElement.style.display === "block") {
        this.noteEditor.domElement.style.display = "none";
        return;
      }

      if (
        event.target instanceof HTMLDivElement &&
        event.target.classList.contains("piano-roll-note")
      ) {
        this.gridElement.classList.add("dragging");
        const note = this.notes.find((n) => n.domElement === event.target);

        if (event.ctrlKey || event.button === 1) {
          if (note)
            this.noteEditor.show(event.clientX + 20, event.clientY - 5, note);
          else this.noteEditor.domElement.style.display = "none";
        }

        if (note) {
          this.currentNote = note;

          if (event.button === 2) {
            this.remove(note);
            this.currentNote = null;
          } else {
            this.currentNote.domElement.parentElement?.appendChild(
              this.currentNote.domElement
            );
            if (x < NOTE_HANDLE_WIDTH) {
              this.dragMode = "start";
            } else if (
              this.currentNote.domElement.offsetWidth - x <
              NOTE_HANDLE_WIDTH
            ) {
              this.dragMode = "end";
            } else {
              this.dragMode = "move";
            }
            console.log(
              "drag mode",
              this.dragMode,
              x,
              this.currentNote.domElement.offsetWidth
            );
          }
        }
      } else if (event.button !== 0) return;
      else if (event.target === this.gridElement) {
        this.gridElement.classList.add("dragging");
        const pitch = 87 - Math.floor(event.layerY / this.noteHeight);

        let start = event.layerX / this.beatWidth;
        start = Math.round(start / this.quantize) * this.quantize;

        this.currentNote = new GridItem(this, {
          pitch: pitch,
          start: start,
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

    this.gridElement.addEventListener("pointermove", (event) => {
      if (this.currentNote && event.buttons === 1) {
        if (this.dragMode === "start") {
          this.currentNote.start = event.layerX / this.beatWidth;
          this.currentNote.start =
            Math.round(this.currentNote.start / this.quantize) * this.quantize;
          this.currentNote.domElement.style.left = `${
            this.currentNote.start * this.beatWidth
          }px`;
          this.currentNote.domElement.style.width = `${
            (this.currentNote.end - this.currentNote.start) * this.beatWidth
          }px`;
        } else if (this.dragMode === "end") {
          this.currentNote.end = event.layerX / this.beatWidth;
          this.currentNote.end =
            Math.round(this.currentNote.end / this.quantize) * this.quantize;
          this.currentNote.domElement.style.width = `${
            (this.currentNote.end - this.currentNote.start) * this.beatWidth
          }px`;
        } else if (this.dragMode === "move") {
          this.currentNote.start =
            (event.layerX - this.dragOffset) / this.beatWidth;
          this.currentNote.start =
            Math.round(this.currentNote.start / this.quantize) * this.quantize;
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
