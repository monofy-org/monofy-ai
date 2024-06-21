import { EventDataMap, IDragEvent, IResizeEvent } from "../EventObject";
import { BaseElement } from "./BaseElement";
import { DraggableWindow } from "./DraggableWindow";
import { WindowSnapPanel } from "./WindowSnapPanel";

export class WindowContainer extends BaseElement<"update"> {
  private _activeWindow: DraggableWindow | null = null;

  readonly windows: DraggableWindow<keyof EventDataMap>[] = [];
  private _draggingWindow: DraggableWindow | null = null;
  private _bottomPanel: WindowSnapPanel;
  private _leftPanel: WindowSnapPanel;
  private _rightPanel: WindowSnapPanel;
  private _workspace: HTMLElement;

  constructor() {
    super("div", "window-container");
    //this.domElement.addEventListener("pointerdown", () => {});

    const topRow = document.createElement("div");
    topRow.classList.add("window-snap-row");
    this.domElement.appendChild(topRow);

    const bottomRow = document.createElement("div");
    bottomRow.classList.add("window-snap-row");
    bottomRow.classList.add("bottom-row");
    this.domElement.appendChild(bottomRow);

    this._leftPanel = new WindowSnapPanel(this);
    topRow.appendChild(this._leftPanel.domElement);

    this._workspace = document.createElement("div");
    this._workspace.classList.add("window-container-workspace");
    topRow.appendChild(this._workspace);

    this._rightPanel = new WindowSnapPanel(this);
    topRow.appendChild(this._rightPanel.domElement);

    this._bottomPanel = new WindowSnapPanel(this);
    bottomRow.appendChild(this._bottomPanel.domElement);
  }

  private _handleWindowDrag(e: IDragEvent) {
    // console.log("drag", e);
    const elt = e.target.domElement;
    const workspaceWidth = this._workspace.getBoundingClientRect().width;
    if (elt.parentElement === this._workspace) {
      const leftPanelWidth =
        this._leftPanel.domElement.getBoundingClientRect().width;
      if (e.event.clientX <= Math.max(0, leftPanelWidth)) {
        const panelWidth = leftPanelWidth || 300;
        this._leftPanel.domElement.appendChild(elt);
        elt.style.width = `${panelWidth}px`;
        elt.style.height = "unset";
        elt.classList.toggle("snap", true);
      } else {
        const rightPanelWidth =
          this._rightPanel.domElement.getBoundingClientRect().width;
        if (
          e.event.clientX + Math.max(250, rightPanelWidth) >
          workspaceWidth + rightPanelWidth + leftPanelWidth
        ) {
          const panelWidth = rightPanelWidth || 300;
          this._rightPanel.domElement.appendChild(elt);
          elt.style.width = `${panelWidth}px`;
          elt.style.height = "unset";
          elt.classList.toggle("snap", true);
        }
      }
    } else {
      if (
        elt.parentElement === this._leftPanel.domElement &&
        (e.event.target as HTMLElement) === this._workspace
      ) {
        this._workspace.appendChild(elt);
        elt.classList.toggle("snap", false);
        elt.style.left = e.event.clientX + "px";
        const win = e.target as DraggableWindow;
        win.resetDragOffset();
        win.setSize(win.options.width!, win.options.height!);
      } else if (
        elt.parentElement === this._rightPanel.domElement &&
        (e.event.target as HTMLElement) === this._workspace
      ) {
        this._workspace.appendChild(elt);
        elt.classList.toggle("snap", false);
        elt.style.left = workspaceWidth + "px";
        const win = e.target as DraggableWindow;
        win.resetDragOffset();
        win.setSize(win.options.width!, win.options.height!);
      }
    }
  }

  private _handleWindowResize(e: IResizeEvent) {
    for (const win of this.windows) {
      if (
        win.domElement !== e.target.domElement &&
        win.domElement.parentElement !== this._workspace &&
        win.domElement.parentElement === e.target.domElement.parentElement
      ) {
        win.domElement.style.width = `${e.width}px`;
      }
    }
  }

  addWindow<T extends DraggableWindow>(window: T) {
    if (this.windows.includes(window)) {
      console.warn("Window already added", window);
      return;
    }
    window.on("drag", (e) => this._handleWindowDrag(e as IDragEvent));
    window.on("resize", (e) => this._handleWindowResize(e as IResizeEvent));
    this.windows.push(window);
    this._workspace.appendChild(window.domElement);
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
