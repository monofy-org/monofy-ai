import { BaseElement } from "./BaseElement";
import { SelectableElement } from "./SelectableElement";

export class SelectableGroup<
  T extends SelectableElement = SelectableElement,
> extends BaseElement<"select", HTMLDivElement> {
  constructor(readonly items: T[] = []) {
    super("div", "selectable-group");
    for (const item of items) {
      this.addSelectable(item, false);
    }
  }

  addSelectable(selectable: T, addToDom = true) {
    if (this.items.indexOf(selectable) !== -1) {
      console.warn("Already added", selectable);
    } else {
      this.items.push(selectable);
      selectable.on("select", () => {
        this.emit("select", this);
      });
      if (addToDom) this.domElement.appendChild(selectable.domElement);

      selectable.domElement.addEventListener("pointerdown", (e) => {        
        if (!e.ctrlKey && !e.shiftKey) {
          for (const item of this.items) {
            item.selected = item === selectable;
          }
        } else if (e.ctrlKey) {
          selectable.selected = !selectable.selected;
        } else if (e.shiftKey) {
          const f = this.items.find((t) => t.selected);
          if (!f) {
            selectable.selected = true;
            return;
          }
          const start = this.items.indexOf(f);
          const end = this.items.indexOf(selectable);
          const min = Math.min(start, end);
          const max = Math.max(start, end);
          for (let i = min; i <= max; i++) {
            this.items[i].selected = true;
          }
        }
      });
    }
  }

  removeSelectable(selectable: T) {
    const index = this.items.indexOf(selectable);
    if (index !== -1) {
      this.items.splice(index, 1);
      this.domElement.removeChild(selectable.domElement);
    }
  }

  clear() {
    for (const item of this.items) {
      this.domElement.removeChild(item.domElement);
    }
    this.items.length = 0;
  }
}
