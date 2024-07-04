import EventObject, { EventDataMap } from "../EventObject";

export abstract class BaseElement<
  T extends keyof EventDataMap = keyof EventDataMap,
  E extends HTMLElement = HTMLElement,
> extends EventObject<T> {
  readonly domElement: E;

  constructor(tagName: string, className?: string) {
    super();
    this.domElement = document.createElement(tagName) as E;
    if (className) {
      this.domElement.classList.add(className);
    }
  }

  dispose() {    
    this.removeAllListeners();
    this.domElement.remove();
  }  

  appendChild(child: BaseElement<keyof EventDataMap>) {
    this.domElement.appendChild(child.domElement);
    return this;
  }

  removeChild(child: BaseElement<keyof EventDataMap>) {
    this.domElement.removeChild(child.domElement);
    return this;
  }
}
