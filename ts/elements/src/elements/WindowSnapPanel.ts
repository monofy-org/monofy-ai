import { BaseElement } from "./BaseElement";
import type { WindowContainer } from "./WindowContainer";

export class WindowSnapPanel extends BaseElement<"open" | "close"> {
  private _isOpen = false;

  get isOpen() {
    return this._isOpen;
  }

  constructor(readonly container: WindowContainer) {
    super("div", "window-snap-panel");

    this.domElement.addEventListener("pointerdown", () => {
      this.toggle();
    });
  }

  open() {
    if (!this._isOpen) {
      this.domElement.classList.toggle("open", true);
      this._isOpen = true;
      this.emit("open");
    }
  }

  close() {
    if (this._isOpen) {
      this.domElement.classList.toggle("open", false);
      this._isOpen = false;
      this.emit("close");
    }
  }

  toggle() {
    if (this._isOpen) {
      this.close();
    } else {
      this.open();
    }
  }
}
