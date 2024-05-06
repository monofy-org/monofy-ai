import { SizableElement } from "./SizableElement";

export class DraggableWindow extends SizableElement<
  "update" | "resize" | "open" | "close"
> {
  readonly titlebar: HTMLElement;
  readonly content: HTMLElement;
  private readonly _title: HTMLElement;
  private readonly _closeButton: HTMLButtonElement;
  private _isDragging = false;
  private _dragOffsetX = 0;
  private _dragOffsetY = 0;
  private _top = 0;
  private _left = 0;

  get isVisible() {
    return this.domElement.style.display !== "none";
  }

  constructor(
    title: string,
    private readonly persistent = false,
    private innerContent: HTMLElement
  ) {
    super("div", "draggable-window");

    this.titlebar = document.createElement("div");
    this.titlebar.className = "window-titlebar";
    this.domElement.appendChild(this.titlebar);

    this.titlebar.addEventListener("pointerdown", (e) => {
      this._isDragging = true;
      this._dragOffsetX = e.layerX;
      this._dragOffsetY = e.layerY;

      this.domElement.style.pointerEvents = "none";

      const ondrag = (e: PointerEvent) => {
        if (this._isDragging) {
          const newTop = e.layerY - this._dragOffsetY;
          const newLeft = e.layerX - this._dragOffsetX;

          this.domElement.style.top = `${newTop}px`;
          this.domElement.style.left = `${newLeft}px`;

          this._top = newTop;
          this._left = newLeft;
        }
      };

      const onrelease = () => {
        this._isDragging = false;
        this.domElement.style.pointerEvents = "all";
        window.removeEventListener("pointermove", ondrag);
        window.removeEventListener("pointerup", onrelease);
      };

      this.domElement.parentElement?.addEventListener("pointermove", ondrag);
      this.domElement.parentElement?.addEventListener("pointerup", onrelease);
    });

    this._title = document.createElement("div");
    this._title.className = "title";
    this._title.textContent = title;
    this.titlebar.appendChild(this._title);

    this.content = document.createElement("div");
    this.content.className = "window-content";
    this.domElement.appendChild(this.content);

    this._closeButton = document.createElement("button");
    this._closeButton.className = "window-close-button";
    this._closeButton.innerHTML = "X";
    this._closeButton.addEventListener("click", this.close.bind(this));

    this.titlebar.appendChild(this._closeButton);

    this.content.appendChild(this.innerContent);
  }

  show(x: number, y: number) {
    this.domElement.style.display = "block";
    this.domElement.style.top = `${y}px`;
    this.domElement.style.left = `${x}px`;
    this.fireEvent("open");
  }

  close() {
    this.domElement.style.display = "none";
    this.fireEvent("close");
    if (!this.persistent) {
      this.domElement.remove();
    }
  }

  setSize(width: number, height: number) {
    this.domElement.style.width = `${width}px`;
    this.domElement.style.height = `${height}px`;
  }

  setPosition(x: number, y: number) {
    this.domElement.style.top = `${y}px`;
    this.domElement.style.left = `${x}px`;
  }
}
