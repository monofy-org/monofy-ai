import { DraggableWindow } from "./DraggableWindow";

export class WindowContainer {
  readonly domElement: HTMLDivElement;

  windows: DraggableWindow[] = [];
  _draggingWindow: DraggableWindow | null = null;

  constructor() {
    this.domElement = document.createElement("div");
    this.domElement.className = "window-container";
    this.domElement.addEventListener("pointerdown", () => {
      
    });
  }

  addWindow(window: DraggableWindow) {
    this.windows.push(window);
    this.domElement.appendChild(window.domElement);
  }
}
