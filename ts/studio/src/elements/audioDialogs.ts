import { LyricCanvas } from "./components/LyricCanvas";
import { GridItem } from "./components/Grid";
import { DialogPopup } from "../../../elements/src/DialogPopup";

export class PianoRollDialog extends DialogPopup {
  domElement: HTMLDivElement;
  closeButton: HTMLButtonElement;
  saveButton: HTMLButtonElement;
  note: GridItem | null = null;

  constructor(public onsave: (note: GridItem) => void) {
    super();

    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll-dialog");

    this.closeButton = document.createElement("button");
    this.closeButton.classList.add("window-close-button");
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

  override show(x: number, y: number, note: GridItem) {
    super.show(x, y, note);
    this.note = note;
    this.domElement.parentElement?.appendChild(this.domElement);
  }
}

export class LyricEditorDialog extends PianoRollDialog {
  textInput: HTMLInputElement;
  audioCanvas: LyricCanvas;
  pitchSlider: HTMLInputElement;
  constructor(onsave: (note: GridItem) => void) {
    super(onsave);
    this.domElement.classList.add("piano-roll-note-editor");
    this.domElement.style.display = "none";

    this.textInput = document.createElement("input");
    this.textInput.setAttribute("placeholder", "Note lyric");
    this.textInput.classList.add("lyric-editor-text");
    this.textInput.type = "text";
    this.domElement.appendChild(this.textInput);

    this.pitchSlider = document.createElement("input");
    this.pitchSlider.classList.add("lyric-editor-pitch");
    this.pitchSlider.type = "range";
    this.pitchSlider.min = "-0.5";
    this.pitchSlider.max = "0.5";
    this.pitchSlider.step = "0.01";
    this.pitchSlider.value = "0";
    this.pitchSlider.addEventListener("input", () => {
      this.note!.note = parseFloat(this.pitchSlider.value);
    });
    this.domElement.appendChild(this.pitchSlider);

    this.audioCanvas = new LyricCanvas();
    this.domElement.appendChild(this.audioCanvas.domElement);

    this.saveButton.addEventListener("click", () => {
      this.audioCanvas.domElement.style.display = "block";
      this.audioCanvas.generateAudio(this.note!, true).then((buffer) => {
        this.note!.audio = buffer;
      });
      this.onsave(this.note!);
    });
  }

  show(x: number, y: number, note: GridItem) {
    super.show(x, y, note);
    this.textInput.value = this.note?.label || "";
    if (this.note?.audio) {
      this.audioCanvas.loadBuffer(this.note.audio);
      this.audioCanvas.domElement.style.display = "block";
    } else {
      this.audioCanvas.domElement.style.display = "none";
    }
  }
}
