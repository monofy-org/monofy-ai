import { LyricCanvas } from "./components/LyricCanvas";
import { DialogPopup } from "../../../elements/src/elements/DialogPopup";
import { IEvent } from "../schema";

export class PianoRollDialog extends DialogPopup {
  domElement: HTMLDivElement;
  closeButton: HTMLButtonElement;
  saveButton: HTMLButtonElement;
  note: IEvent | null = null;

  constructor(public onsave: (note: IEvent) => void) {
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

  override show(x: number, y: number, note: IEvent) {
    super.show(x, y, note);
    this.note = note;
    this.domElement.parentElement?.appendChild(this.domElement);
  }
}

export class LyricEditorDialog extends PianoRollDialog {
  textInput: HTMLInputElement;
  audioCanvas: LyricCanvas;  
  constructor(onsave: (note: IEvent) => void) {
    super(onsave);
    this.domElement.classList.add("grid-item-editor");
    this.domElement.style.display = "none";

    this.textInput = document.createElement("input");
    this.textInput.setAttribute("placeholder", "Note lyric");
    this.textInput.classList.add("lyric-editor-text");
    this.textInput.type = "text";
    this.domElement.appendChild(this.textInput);

    this.audioCanvas = new LyricCanvas();
    this.domElement.appendChild(this.audioCanvas.domElement);

    this.saveButton.addEventListener("click", () => {
      this.audioCanvas.domElement.style.display = "block";
      // this.audioCanvas.generateAudio(this.note!, true).then((buffer) => {
      //   this.note!.audio = buffer;
      // });
      // this.onsave(this.note!);
    });
  }

  show(x: number, y: number, event: IEvent) {
    super.show(x, y, event);
    this.textInput.value = this.note?._label || "";
    // if (this.note?.audio) {
    //   this.audioCanvas.loadBuffer(this.note.audio);
    //   this.audioCanvas.domElement.style.display = "block";
    // } else {
    //   this.audioCanvas.domElement.style.display = "none";
    // }
  }
}
