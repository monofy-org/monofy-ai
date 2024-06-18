import { EventDataMap } from "../EventObject";
import { DraggableWindow } from "./DraggableWindow";

export class WindowContainer {
  readonly domElement: HTMLDivElement;
  private _activeWindow: DraggableWindow | null = null;

  windows: DraggableWindow<keyof EventDataMap>[] = [];
  _draggingWindow: DraggableWindow | null = null;

  constructor() {
    this.domElement = document.createElement("div");
    this.domElement.className = "window-container";
    //this.domElement.addEventListener("pointerdown", () => {});
  }

  addWindow<T extends DraggableWindow>(window: T) {
    if (this.windows.includes(window)) {
      console.warn("Window already added", window);
      return;
    }
    this.windows.push(window);
    this.domElement.appendChild(window.domElement);
    if (!this._activeWindow) {
      this._activeWindow = window;
      this.activateWindow(window);
    }

    window.domElement.addEventListener("pointerdown", () => {
      this.activateWindow(window);
    });

    window.on("open", () => {
      this.activateWindow(window);
    });

    window.on("close", () => {
      if (this._activeWindow === window) {
        this._activeWindow = null;
      }
    });
  }

  activateWindow(window: DraggableWindow) {
    if (this._activeWindow === window) {
      return;
    }

    this._activeWindow = window;

    for (const w of this.windows) {
      w.domElement.classList.toggle("active", w === window);
      w.domElement.style.zIndex = w === window ? "1" : "0";
    }
  }
}
