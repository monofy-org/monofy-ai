import { EventDataMap } from "../EventObject";
import { BaseElement } from "./BaseElement";

export abstract class ScrollPanel<
  T extends keyof EventDataMap,
> extends BaseElement<"scroll" | T> {
  sensitivity = 50;
  private linkedElements: HTMLElement[] = [];
  private _scrollTop = 0;

  get scrollTop() {
    return this._scrollTop;
  }

  set scrollTop(value: number) {
    const top = -value;
    this._scrollTop = top;
    this.scrollElement.style.transform = `translateY(${top}px)`;
    for (const linkedElement of this.linkedElements) {
      linkedElement.style.transform = `translateY(${top}px)`;
    }
  }

  constructor(readonly scrollElement: HTMLElement) {
    super("div", "scroll-panel");
    scrollElement.classList.add("scroll-panel-content");

    this.scrollElement.addEventListener("wheel", (e) => {
      e.preventDefault();
      let top =
        this._scrollTop + (e.deltaY > 0 ? -this.sensitivity : this.sensitivity);
      const bottom = top + this.scrollElement.offsetHeight;
      if (bottom < this.domElement.offsetHeight) {
        top = this.domElement.offsetHeight - this.scrollElement.offsetHeight;
      } else if (top > 0) {
        top = 0;
      }
      this._scrollTop = top;
      this.scrollElement.style.transform = `translateY(${top}px)`;
      for (const linkedElement of this.linkedElements) {
        linkedElement.style.transform = `translateY(${top}px)`;
      }

      this.emit("scroll", e);
    });
  }

  linkElement(element: HTMLElement) {
    this.linkedElements.push(element);
    element.classList.add("scroll-panel-content");
  }
}
