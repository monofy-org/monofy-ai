import { EventDataMap } from "../EventObject";
import { SizableElement } from "./SizableElement";

export class DraggableWindow<
  T extends keyof EventDataMap,
> extends SizableElement<"update" | "resize" | "open" | "close" | T> {
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

    this.domElement.addEventListener("pointerdown", () => {
      this.domElement.parentElement!.appendChild(this.domElement);
    });

    this.titlebar = document.createElement("div");
    this.titlebar.className = "window-titlebar";
    this.domElement.appendChild(this.titlebar);

    this.titlebar.addEventListener("pointerdown", (e) => {
      this._isDragging = true;
      this._dragOffsetX =
        e.clientX - this.domElement.getBoundingClientRect().left;
      this._dragOffsetY =
        e.clientY -
        this.domElement.getBoundingClientRect().top +
        this.domElement.parentElement!.getBoundingClientRect().top +
        this.domElement.parentElement!.scrollTop;

      const ondrag = (e: PointerEvent) => {
        if (this._isDragging) {
          console.log("Dragging");

          let newTop = e.clientY - this._dragOffsetY;
          let newLeft = e.clientX - this._dragOffsetX;

          if (newTop < 0) {
            newTop = 0;
          }

          if (newLeft < 0) {
            newLeft = 0;
          }

          this.domElement.style.top = `${newTop}px`;
          this.domElement.style.left = `${newLeft}px`;

          this._top = newTop;
          this._left = newLeft;
        }
      };

      const onrelease = () => {
        this._isDragging = false;
        document.body.removeEventListener("pointermove", ondrag);
        document.body.removeEventListener("pointerup", onrelease);
      };

      document.body.addEventListener("pointermove", ondrag);
      document.body.addEventListener("pointerup", onrelease);
    });

    this._title = document.createElement("div");
    this._title.className = "window-titlebar-title";
    this._title.textContent = title;
    this.titlebar.appendChild(this._title);

    this.content = document.createElement("div");
    this.content.className = "window-content";
    this.domElement.appendChild(this.content);

    this._closeButton = document.createElement("button");
    this._closeButton.className = "window-close-button";
    this._closeButton.innerHTML = "X";
    this._closeButton.addEventListener("pointerdown", () => {
      this.close();
    });

    this.titlebar.appendChild(this._closeButton);

    this.content.appendChild(this.innerContent);
  }

  show(x?: number, y?: number) {
    this.domElement.style.display = "flex";
    if (y) this.domElement.style.top = `${y}px`;
    if (x) this.domElement.style.left = `${x}px`;
    setTimeout(() => {
      this.domElement.parentElement?.appendChild(this.domElement);
      this.fireEvent("open");
    }, 1);    
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

  setTitle(title: string) {
    this._title.textContent = title;
  }
}
