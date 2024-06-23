import { EventDataMap } from "../EventObject";
import { BaseElement } from "./BaseElement";
import { SelectableGroup } from "./SelectableGroup";

export class SelectableElement<
  T extends keyof EventDataMap = keyof EventDataMap,
> extends BaseElement<"select" | T> {
  get selected() {
    return this.domElement.classList.contains("selected");
  }

  set selected(value: boolean) {
    if (value === this.selected) return;
    this.domElement.classList.toggle("selected", value);
    this.emit("select", this);
  }

  constructor(
    readonly group: SelectableGroup<SelectableElement>,
    tagName: string,
    className?: string
  ) {
    super(tagName, className);
    this.group.addSelectable(this);
  }
}
