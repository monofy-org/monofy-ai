import { AudioCanvas } from "./AudioCanvas";
import { GridItem } from "./Grid";

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

export class LyricEditorDialog extends PianoRollDialog {
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

    this.audioCanvas = new AudioCanvas();
    this.domElement.appendChild(this.audioCanvas.domElement);

    this.saveButton.addEventListener("click", () => {
      this.audioCanvas.domElement.style.display = "block";
      this.audioCanvas.generateAudio(this.note!, true).then((buffer) => {
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
