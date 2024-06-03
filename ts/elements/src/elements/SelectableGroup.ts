import { BaseElement } from "./BaseElement";
import { SelectableElement } from "./SelectableElement";

export class SelectableGroup<
  T extends SelectableElement,
> extends BaseElement<"select"> {
  readonly selected: T[];

  constructor(readonly items: T[]) {
    super("div", "selectable-group");
    this.selected = [];
    this.items.forEach(this.add.bind(this));
  }

  add(selectable: T) {
    if (this.selected.indexOf(selectable) !== -1) {
      console.warn("Already added", selectable);
    } else {
      selectable.on("select", () => {
        for (const item of this.selected) {
          item.domElement.classList.toggle("selected", item === selectable);
        }
        if (selectable.selected) {
          this.selected.push(selectable);
        }
        this.emit("select", this);
      });
      this.domElement.appendChild(selectable.domElement);
    }
  }

  remove(selectable: T) {
    const index = this.items.indexOf(selectable);
    if (index !== -1) {
      this.items.splice(index, 1);
      this.domElement.removeChild(selectable.domElement);
    }
  }
}
