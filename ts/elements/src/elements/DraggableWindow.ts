import EventObject from "../EventObject";

export class DraggableWindow extends EventObject<
  "update" | "resize" | "open" | "close"
> {
  domElement: HTMLElement;
  titlebar: HTMLElement;
  title: HTMLElement;
  private _isDragging = false;
  private _dragOffsetX = 0;
  private _dragOffsetY = 0;
  private _closeButton: HTMLButtonElement;
  private _top = 0;
  private _left = 0;
  content: HTMLElement;

  get isVisible() {
    return this.domElement.style.display !== "none";
  }

  constructor(
    title: string,
    private readonly persistent = false,
    private innerContent: HTMLElement
  ) {
    super();
    this.domElement = document.createElement("div");
    this.domElement.className = "draggable-window";

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
          this.domElement.style.transform = `translate(${
            e.layerX - this._dragOffsetX
          }px, ${e.layerY - this._dragOffsetY}px)`;

          this._top = e.layerX - this._dragOffsetY;
          this._left = e.layerY - this._dragOffsetX;
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

    this.title = document.createElement("div");
    this.title.className = "title";
    this.title.textContent = title;
    this.titlebar.appendChild(this.title);

    this.content = document.createElement("div");
    this.content.className = "window-content";
    this.domElement.appendChild(this.content);

    this._closeButton = document.createElement("button");
    this._closeButton.className = "close-button";
    this._closeButton.innerHTML = "X";
    this._closeButton.addEventListener("click", this.close.bind(this));

    this.titlebar.appendChild(this._closeButton);

    this.content.appendChild(innerContent);
  }

  show(x: number, y: number) {
    this.domElement.style.display = "block";
    this.domElement.style.transform = `translate(${x}px, ${y}px)`;
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
    this.domElement.style.transform = `translate(${x}px, ${y}px)`;
  }
}
