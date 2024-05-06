import { EventDataMap } from "../EventObject";
import { BaseElement } from "./BaseElement";

export abstract class SizableElement<
  T extends keyof EventDataMap,
> extends BaseElement<"resize" | T> {
  private _resizing = false;
  private _startX = 0;
  private _startY = 0;
  private _startWidth = 0;
  private _startHeight = 0;
  private _startTop = 0;
  private _startLeft = 0;

  private _resizeDirection: "top" | "right" | "bottom" | "left" | null = null;

  constructor(tagName: string, className?: string) {
    super(tagName, className);

    this.domElement.addEventListener("pointermove", (e) => {
      if (this._resizing) {
        const dx = e.clientX - this._startX;
        const dy = e.clientY - this._startY;

        if (this._resizeDirection === "right") {
          this.domElement.style.width = `${this._startWidth + dx}px`;
        } else if (this._resizeDirection === "bottom") {
          this.domElement.style.height = `${this._startHeight + dy}px`;
        } else if (this._resizeDirection === "left") {
          this.domElement.style.width = `${this._startWidth - dx}px`;
          this.domElement.style.left = `${this._startLeft + dx}px`;
        } else if (this._resizeDirection === "top") {
          this.domElement.style.height = `${this._startHeight - dy}px`;
          this.domElement.style.top = `${this._startTop + dy}px`;
        }

        this.fireEvent("resize", {
          width: this.domElement.offsetWidth,
          height: this.domElement.offsetHeight,
        });
      } else {
        const handle = this.getCurrentHandle(e);
        if (handle === "top" || handle === "bottom") {
          this.domElement.style.cursor = "ns-resize";
        } else if (handle === "left" || handle === "right") {
          this.domElement.style.cursor = "ew-resize";
        } else {
          this.domElement.style.cursor = "auto";
        }
      }
    });

    this.domElement.addEventListener("pointerdown", (e) => {
      const handle = this.getCurrentHandle(e);
      if (handle) {
        this.startResize(e, handle);
      }
      const onrelease = () => {
        this.stopResize();
        window.removeEventListener("pointerup", onrelease);
      };

      window.addEventListener("pointerup", onrelease);
    });
  }

  protected startResize(
    e: PointerEvent,
    direction: "top" | "right" | "bottom" | "left"
  ) {
    this._resizing = true;
    this.domElement.style.pointerEvents = "none";
    this._startX = e.clientX;
    this._startY = e.clientY;
    this._startWidth = this.domElement.offsetWidth;
    this._startHeight = this.domElement.offsetHeight;
    this._startTop = this.domElement.offsetTop;
    this._startLeft = this.domElement.offsetLeft;
    this._resizeDirection = direction;
  }

  protected stopResize() {
    this._resizing = false;
    this.domElement.style.pointerEvents = "all";
  }

  getCurrentHandle(e: PointerEvent) {
    const rect = this.domElement.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;

    if (offsetX > 0 && offsetX < 10) {
      return "left";
    } else if (rect.width - offsetX < 10) {
      return "right";
    } else if (offsetY > 0 && offsetY < 10) {
      return "top";
    } else if (rect.height - offsetY < 10) {
      return "bottom";
    }
    return null;
  }
}
