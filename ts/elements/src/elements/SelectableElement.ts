import { EventDataMap } from "../EventObject";
import { BaseElement } from "./BaseElement";

export abstract class SelectableElement<
  T extends keyof EventDataMap = "select",
> extends BaseElement<"select" | T> {
  get selected() {
    return this.domElement.classList.contains("selected");
  }

  constructor(tagName: string, className?: string) {
    super(tagName, className);

    this.domElement.addEventListener("pointerdown", () => {
      this.domElement.classList.toggle("selected");
      this.emit("select", this);
    });
  }
}
