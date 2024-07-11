import { DialogPopup } from "../../../elements/src/elements/DialogPopup";
import { IEvent } from "../schema";

export class PianoRollDialog extends DialogPopup {
  domElement: HTMLDivElement;
  note: IEvent | null = null;

  constructor(public onsave: (note: IEvent) => void) {
    super();

    this.domElement = document.createElement("div");
    this.domElement.classList.add("piano-roll-dialog");
  }

  override show(x: number, y: number, note: IEvent) {
    super.show(x, y, note);
    this.note = note;
    this.domElement.parentElement?.appendChild(this.domElement);
  }
}

export class EventEditorDialog extends PianoRollDialog {
  constructor(onsave: (note: IEvent) => void) {
    super(onsave);
    this.domElement.classList.add("grid-item-editor");
    this.domElement.style.display = "none";
  }
}
