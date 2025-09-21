import { PopupDialog } from "../../../elements/src/elements/PopupDialog";
import { IEvent } from "../schema";

export class PianoRollDialog extends PopupDialog {
  domElement: HTMLDivElement;
  note: IEvent | null = null;

  constructor(public onsave: (note: IEvent) => void) {
    const domElement = document.createElement("div");
    super(domElement);

    this.domElement = domElement;
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
