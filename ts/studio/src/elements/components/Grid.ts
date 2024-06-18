import { ScrollPanel } from "../../../../elements/src/elements/ScrollPanel";
import { DEFAULT_NOTE_HEIGHT, NOTE_NAMES } from "../../constants";
import { LyricEditorDialog } from "../audioDialogs";
import { IEvent, ISequence } from "../../schema";

function getNoteNameFromPitch(pitch: number): string {
  const note = NOTE_NAMES[pitch % NOTE_NAMES.length];
  return `${note}${Math.floor(pitch / NOTE_NAMES.length)}`;
}

const NOTE_HANDLE_WIDTH = 6;

export class GridItem implements IEvent {
  note: number;
  start: number;
  duration: number;
  velocity: number = 0.8;
  labelElement: HTMLDivElement;
  label: string | "" = "";
  audio: AudioBuffer | null = null;
  domElement: HTMLDivElement;

  constructor(
    private readonly grid: Grid,
    item: IEvent,
    label?: string,
    image?: string
  ) {
    const noteItem = item as IEvent;

    this.domElement = document.createElement("div");
    this.domElement.classList.add("grid-item");
    this.domElement.style.height = `${grid.rowHeight}px`;

    this.labelElement = document.createElement("div");
    this.labelElement.classList.add("grid-item-label");
    this.domElement.appendChild(this.labelElement);

    if (image) {
      const image = document.createElement("div");
      image.classList.add("grid-item-image");
      image.style.backgroundImage = `url(${image})`;
      this.domElement.appendChild(image);
      this.domElement.classList.add("has-image");
    }

    if (noteItem.velocity) this.velocity = noteItem.velocity;
    if (noteItem.note) {
      this.note = noteItem.note;
      this.labelElement.textContent = label || getNoteNameFromPitch(this.note);
    } else {
      this.note = 0;
    }

    this.start = item.start;
    this.duration = item.duration;

    if (item.label) this.labelElement.textContent = item.label;

    console.log("GridItem", this);

    grid.gridElement.appendChild(this.domElement);
  }
}

export class Grid extends ScrollPanel<"select" | "update" | "release"> {
  readonly domElement: HTMLDivElement;
  readonly gridElement: HTMLDivElement;
  readonly previewCanvas: HTMLCanvasElement;
  readonly noteEditor: LyricEditorDialog;
  private _currentItem: IEvent | null = null;
  private _dragMode: string | null = null;
  private _dragOffset: number = 0;
  private _track: ISequence | null = null;
  readonly scrollElement: HTMLDivElement;

  public quantize: number = 0.25;
  public drawingEnabled: boolean = true;
  public drawingLabel: string | undefined = undefined;
  public drawingImage: string | undefined = undefined;

  constructor(
    readonly rowCount = 88,
    readonly beatWidth = 100,
    readonly rowHeight = DEFAULT_NOTE_HEIGHT
  ) {
    const gridElement = document.createElement("div");

    super(gridElement);

    this.scrollElement = gridElement;

    this.domElement = document.createElement("div");

    this.gridElement = gridElement;

    this.domElement.classList.add("grid-container");
    this.gridElement.classList.add("grid");

    this.gridElement.addEventListener("dragstart", (event) => {
      event.preventDefault();
    });

    this.previewCanvas = document.createElement("canvas");
    this.previewCanvas.style.imageRendering = "pixelated";

    const rowBackgroundCanvas = document.createElement("canvas");
    rowBackgroundCanvas.style.imageRendering = "pixelated";
    rowBackgroundCanvas.width = this.beatWidth * 4;
    rowBackgroundCanvas.height = this.rowHeight;
    console.log("debug", rowBackgroundCanvas.width, rowBackgroundCanvas.height);
    const ctx = rowBackgroundCanvas.getContext("2d");
    // disable anti-alias

    if (ctx) {
      ctx.clearRect(
        0,
        0,
        rowBackgroundCanvas.width,
        rowBackgroundCanvas.height
      );
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#444";
      for (let j = 1; j < 4; j++) {
        ctx.beginPath();
        ctx.moveTo(j * this.beatWidth, 0);
        ctx.lineTo(j * this.beatWidth, this.rowHeight);
        ctx.stroke();
      }
      ctx.strokeStyle = "#999";
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(0, this.rowHeight);
      ctx.stroke();
    }
    for (let i = 0; i < rowCount; i++) {
      const row = document.createElement("div");
      row.classList.add("grid-row");
      row.style.height = `${this.rowHeight}px`;
      this.gridElement.appendChild(row);
      const base64 = rowBackgroundCanvas.toDataURL();
      row.style.backgroundImage = `url(${base64})`;
      row.style.backgroundRepeat = "repeat";
    }
    this.domElement.appendChild(this.gridElement);

    this.noteEditor = new LyricEditorDialog((note) => {
      const input = this.noteEditor.domElement.querySelector(
        ".lyric-editor-text"
      ) as HTMLInputElement;
      note.label = input?.value || "";
    });

    document.body.appendChild(this.noteEditor.domElement);

    this.gridElement.addEventListener("pointerdown", (event) => {
      if (!this._track) throw new Error("No track");

      if (!this._track.events) throw new Error("No events");

      if (!this.drawingEnabled) return;

      event.preventDefault();

      this._dragOffset = event.layerX;

      if (this.noteEditor.domElement.style.display === "block") {
        this.noteEditor.domElement.style.display = "none";
        return;
      }

      if (
        event.target instanceof HTMLDivElement &&
        event.target.classList.contains("grid-item")
      ) {
        this.gridElement.classList.add("dragging");
        const note = this._track.events.find(
          (n) => n.domElement === event.target
        );

        if (event.ctrlKey || event.button === 1) {
          if (note)
            this.noteEditor.show(event.clientX + 20, event.clientY - 5, note);
          else this.noteEditor.domElement.style.display = "none";
        }

        if (note) {
          this._currentItem = note;

          if (event.button === 2) {
            this.remove(note);
            this._currentItem = null;
          } else {
            note.domElement!.parentElement?.appendChild(note.domElement!);
            note.domElement!.style.width =
              note.duration * this.beatWidth + "px";
            if (this._dragOffset < NOTE_HANDLE_WIDTH) {
              this._dragMode = "start";
            } else if (
              note.domElement!.offsetWidth - this._dragOffset <
              NOTE_HANDLE_WIDTH
            ) {
              this._dragMode = "end";
            } else {
              this._dragMode = "move";
            }
          }
        }
      } else if (event.button !== 0) return;
      else if (event.target === this.gridElement) {
        this.gridElement.classList.add("dragging");
        const pitch = 87 - Math.floor(event.layerY / this.rowHeight);

        let start = event.layerX / this.beatWidth;
        start = Math.round(start / this.quantize) * this.quantize;
        console.log("start", start, event.layerX);

        const item: IEvent = {
          note: pitch,
          start: start,
          velocity: 100,
          duration: this.quantize,
          label: "",
          domElement: undefined,
        };

        this._currentItem = this.add(item);
        this._currentItem.domElement!.style.top = `${
          (87 - this._currentItem.note!) * this.rowHeight
        }px`;
        this._currentItem.domElement!.style.left = `${
          this._currentItem.start * this.beatWidth
        }px`;
        this._dragMode = "end";

        this.emit("select", this._currentItem);
      }

      this.emit("update", this);
    });

    this.gridElement.addEventListener("pointerup", () => {
      this.gridElement.classList.remove("dragging");

      this.emit("update", this);
      this.emit("release", this._currentItem);

      this._currentItem = null;
    });

    this.gridElement.addEventListener("pointerleave", () => {
      this.gridElement.classList.remove("dragging");
      if (this._currentItem) {
        // this.remove(this._currentItem);
        this._currentItem = null;
      }
    });

    this.gridElement.addEventListener("contextmenu", (event) => {
      event.preventDefault();
    });

    this.gridElement.addEventListener("pointermove", (event) => {
      if (this._currentItem && event.buttons === 1) {
        if (this._dragMode === "start") {
          const start =
            Math.round(event.layerX / this.beatWidth / this.quantize) *
            this.quantize;

          const oldStart = this._currentItem.start;

          if (start >= this._currentItem.start + this._currentItem.duration) {
            return;
          }

          this._currentItem.start = start;

          this._currentItem.duration =
            oldStart - this._currentItem.start + this._currentItem.duration;
          this._currentItem.domElement!.style.left = `${
            this._currentItem.start * this.beatWidth
          }px`;
          this._currentItem.domElement!.style.width = `${Math.max(
            this.quantize,
            this._currentItem.duration * this.beatWidth
          )}px`;
        } else if (this._dragMode === "end") {
          this._currentItem.duration =
            event.layerX / this.beatWidth - this._currentItem.start;

          // quantize
          this._currentItem.duration =
            Math.round(this._currentItem.duration / this.quantize) *
            this.quantize;

          this._currentItem.domElement!.style.width = `${Math.max(
            this.quantize,
            this._currentItem.duration * this.beatWidth
          )}px`;
        } else if (this._dragMode === "move") {
          this._currentItem.start =
            Math.max(0, event.layerX - this._dragOffset) / this.beatWidth;
          this._currentItem.start =
            Math.round(this._currentItem.start / this.quantize) * this.quantize;
          this._currentItem.domElement!.style.left = `${
            this._currentItem.start * this.beatWidth
          }px`;
        } else {
          return;
        }

        this.emit("update", this);
      }
    });

    this.gridElement.addEventListener("pointerup", () => {
      this.gridElement.classList.remove("dragging");
      if (this._currentItem) {
        this._currentItem.domElement!.style.pointerEvents = "auto";
      }
      this._currentItem = null;
    });

    // this.gridElement.addEventListener("pointerleave", () => {
    //   this.gridElement.classList.remove("dragging");
    //   if (this._currentItem) {
    //     this._currentItem.domElement!.style.pointerEvents = "auto";
    //   }
    //   this._currentItem = null;
    // });
  }

  add(event: IEvent) {
    if (!this._track) throw new Error("add() No track!");
    const item = new GridItem(
      this,
      event,
      this.drawingLabel,
      this.drawingImage
    );
    this._track.events.push(item);
    event.domElement = item.domElement;
    console.log("Grid: add", event);
    return item;
  }

  remove(event: IEvent) {
    if (!this._track) throw new Error("remove() No track!");
    this._track.events.splice(this._track.events.indexOf(event), 1);
    this.gridElement.removeChild(event.domElement!);
  }

  load(track: ISequence) {
    if (!track) {
      throw new Error("load() No track!");
    }
    console.log("Grid: load", track.events);
    for (const event of this._track?.events || []) {
      event.domElement?.remove();
    }
    this._track = track;
    for (const event of track.events) {
      this.gridElement.appendChild(event.domElement!);
    }
  }
}
