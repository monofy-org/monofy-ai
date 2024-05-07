import { EventDataMap } from "../EventObject";
import { DraggableWindow } from "./DraggableWindow";

export class WindowContainer {
  readonly domElement: HTMLDivElement;

  windows: DraggableWindow<keyof EventDataMap>[] = [];
  _draggingWindow: DraggableWindow<keyof EventDataMap> | null = null;

  constructor() {
    this.domElement = document.createElement("div");
    this.domElement.className = "window-container";
    this.domElement.addEventListener("pointerdown", () => {});
  }

  addWindow<T extends DraggableWindow<keyof EventDataMap>>(window: T) {
    this.windows.push(window);
    this.domElement.appendChild(window.domElement);
  }
}
