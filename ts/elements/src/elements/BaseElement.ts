import EventObject, { EventDataMap } from "../EventObject";

export abstract class BaseElement<
  T extends keyof EventDataMap,
> extends EventObject<T> {
  domElement: HTMLElement;

  constructor(tagName: string, className?: string) {
    super();
    this.domElement = document.createElement(tagName);
    if (className) {
      this.domElement.classList.add(className);
    }
  }
}
