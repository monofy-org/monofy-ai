import { ScrollPanel } from "../../../../elements/src/elements/ScrollPanel";
import { DEFAULT_NOTE_HEIGHT, NOTE_NAMES } from "../../constants";
import { LyricEditorDialog } from "../audioDialogs";
import { IEvent, ISequence } from "../../schema";
import { SelectableElement } from "../../../../elements/src/elements/SelectableElement";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";

function getNoteNameFromPitch(pitch: number): string {
  const note = NOTE_NAMES[pitch % NOTE_NAMES.length];
  return `${note}${Math.floor(pitch / NOTE_NAMES.length)}`;
}

const NOTE_HANDLE_WIDTH = 6;

export class GridItem extends SelectableElement implements IEvent {
  labelElement: HTMLDivElement;
  _label: string | "" = "";

  get label() {
    return this._label;
  }

  get note() {
    return this.event.note || 0;
  }

  set note(value: number) {
    this.event.note = value;
  }

  get start() {
    return this.event.start;
  }

  set start(value: number) {
    this.event.start = value;
  }

  get duration() {
    return this.event.duration;
  }

  set duration(value: number) {
    this.event.duration = value;
  }

  get velocity() {
    return this.event.velocity || 0;
  }

  set velocity(value: number) {
    this.event.velocity = value;
  }

  set label(value: string) {
    this._label = value;
    this.labelElement.textContent = value;
  }

  constructor(
    private readonly grid: Grid,
    readonly event: IEvent,
    label?: string,
    src?: string,
    readonly value?: unknown
  ) {
    super("div", "grid-item");
    const noteItem = event as IEvent;

    this.domElement.style.height = `${grid.rowHeight}px`;

    this.labelElement = document.createElement("div");
    this.labelElement.classList.add("grid-item-label");
    this.domElement.appendChild(this.labelElement);

    if (src) {
      const image = document.createElement("div");
      image.classList.add("grid-item-image");
      console.log(
        "DEBUG Background image",
        src,
        image.style.backgroundImage,
        `url(${src})`
      );
      image.style.backgroundImage = `url(${src})`;
      this.domElement.appendChild(image);
      this.domElement.classList.add("has-image");
    }

    if (noteItem.velocity) this.velocity = noteItem.velocity;
    if (noteItem.note) {
      this.note = noteItem.note;
      this._label = label || getNoteNameFromPitch(this.note);
      this.labelElement.textContent = this._label;
    } else {
      this.note = 0;
    }

    this.start = event.start;
    this.duration = event.duration;

    if (event._label) this.labelElement.textContent = event._label;

    console.log("GridItem", this);

    grid.gridElement.appendChild(this.domElement);
  }
}

export class Grid extends ScrollPanel<
  "select" | "update" | "release" | "add" | "remove"
> {
  readonly gridElement: HTMLDivElement;
  readonly previewCanvas: HTMLCanvasElement;
  readonly noteEditor: LyricEditorDialog;
  private _currentItem: GridItem | null = null;
  private _dragMode: string | null = null;
  private _dragOffset: number = 0;
  private _sequence: ISequence | null = null;
  private _items: Map<IEvent, GridItem> = new Map();
  readonly scrollElement: HTMLDivElement;

  public quantize: number = 0.25;
  public drawingEnabled: boolean = true;
  public drawingLabel: string | undefined = undefined;
  public drawingImage: string | undefined = undefined;
  public drawingValue: unknown | undefined = undefined;

  constructor(
    readonly rowCount = 88,
    readonly beatWidth = 100,
    readonly rowHeight = DEFAULT_NOTE_HEIGHT
  ) {
    const grid = new SelectableGroup<GridItem>();

    super(grid.domElement);

    this.scrollElement = grid.domElement;

    this.gridElement = grid.domElement;

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
      note._label = input?.value || "";
    });

    document.body.appendChild(this.noteEditor.domElement);

    this.gridElement.addEventListener("pointerdown", (e) => {
      if (!this._sequence) throw new Error("No track");

      if (!this._sequence.events) throw new Error("No events");

      if (!this.drawingEnabled) return;

      e.preventDefault();

      this._dragOffset = e.layerX;

      if (this.noteEditor.domElement.style.display === "block") {
        this.noteEditor.domElement.style.display = "none";
        return;
      }

      if (
        e.target instanceof HTMLDivElement &&
        e.target.classList.contains("grid-item")
      ) {
        this.gridElement.classList.add("dragging");
        const note = Array.from(this._items.values()).find(
          (item: GridItem) => item.domElement === e.target
        );

        if (e.ctrlKey || e.button === 1) {
          if (note) this.noteEditor.show(e.clientX + 20, e.clientY - 5, note);
          else this.noteEditor.domElement.style.display = "none";
        }

        if (note) {
          this._currentItem = note;

          if (e.button === 2) {
            this.remove(note.event);
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
      } else if (e.button !== 0) return;
      else if (e.target === this.gridElement) {
        this.gridElement.classList.add("dragging");
        const row = Math.floor(e.layerY / this.rowHeight);
        const note = 87 - row;

        let start = e.layerX / this.beatWidth;
        start = Math.round(start / this.quantize) * this.quantize;
        console.log("start", start, e.layerX);

        const item: IEvent = {
          row: row,
          note: note,
          start: start,
          velocity: 1,
          duration: this.quantize,
          _label: "",
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

  add(item: IEvent) {
    if (!this._sequence) throw new Error("add() No sequence loaded!");
    const gridItem = new GridItem(
      this,
      item,
      this.drawingLabel,
      this.drawingImage,
      this.drawingValue
    );
    this._sequence.events.push(item);
    this._items.set(item, gridItem);
    this.emit("add", gridItem);
    return gridItem;
  }

  remove(item: IEvent) {
    if (!this._sequence) throw new Error("remove() No sequence loaded!");
    console.log("Grid: remove", item);
    const gridItem = this._items.get(item);
    if (!gridItem) {
      console.error("remove() Item not found!", item, this._items);
      return;
    }
    gridItem.domElement.remove();
    this._items.delete(item);
    this._sequence.events.splice(this._sequence.events.indexOf(item), 1);
    this.emit("remove", item);
  }

  load(sequence: ISequence) {
    if (!sequence) {
      throw new Error("load() No track!");
    }
    console.log("Grid: load", sequence.events);

    this._items.clear();

    const items = this.gridElement.querySelectorAll(".grid-item");
    items.forEach((item) => item.remove());

    this._sequence = sequence;

    for (const event of sequence.events) {
      this.add(event);
    }
  }
}
