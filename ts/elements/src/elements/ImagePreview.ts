import { BaseElement } from "./BaseElement";

export class ImagePreview extends BaseElement<"change"> {
  private _img: HTMLImageElement;
  private zoom = 1;

  constructor(src?: string | File) {
    super("div", "image-preview");

    this._img = document.createElement("img");
    this._img.addEventListener("load", () => {
      this.emit("change", this);
    });

    if (src instanceof File) {
      this._img.src = URL.createObjectURL(src);
    } else if (typeof src === "string") {
      this._img.src = src;
    }

    this.domElement.appendChild(this._img);

    // Add pinch-zoom and wheel zoom event listeners
    this.domElement.addEventListener("wheel", this.handleWheelZoom);
    this.domElement.addEventListener("touchstart", this.handlePinchZoom);
    this.domElement.addEventListener("touchmove", this.handlePinchZoom);
  }

  set src(value: string) {
    this._img.src = value;
  }

  private handleWheelZoom = (e: WheelEvent) => {
    e.preventDefault();
    const dir = e.deltaY > 0 ? -1 : 1;
    this.zoom *= 1 + dir * 0.1;
    if (this.zoom < 0.1) {
      this.zoom = 0.1;
    }
    this._img.style.transform = `scale(${this.zoom})`;
  };

  private handlePinchZoom = (event: TouchEvent) => {
    const touch1 = event.touches[0];
    const touch2 = event.touches[1];
    if (!touch1 || !touch2) {
      return;
    }
    const distance = Math.hypot(
      touch1.clientX - touch2.clientX,
      touch1.clientY - touch2.clientY
    );
    this._img.style.transform = `scale(${distance * 0.01})`;
  };
}
