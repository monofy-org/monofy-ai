import { CanvasHistory } from "./CanvasHistory";

class MultiCanvas {
  history: CanvasHistory;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  cursor = document.createElement("div");
  drawing = false;
  brushSize: number = 3;  

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    this.history = new CanvasHistory(this.ctx);
    this.cursor = document.createElement("div");
    this.cursor.style.position = "absolute";
    this.cursor.style.background = "black";
    this.cursor.style.borderRadius = "50%";
    this.cursor.style.width = this.brushSize * 2 + "px";
    this.cursor.style.height = this.brushSize * 2 + "px";
    this.cursor.style.pointerEvents = "none";
    this.canvas.parentElement!.appendChild(this.cursor);
    this.canvas.addEventListener("mousemove", (e: MouseEvent) => {
      this.cursor.style.left = e.offsetX - this.brushSize + "px";
      this.cursor.style.top = e.offsetY - this.brushSize + "px";
      if (this.drawing) {
        this.ctx.beginPath();
        this.ctx.arc(e.offsetX, e.offsetY, this.brushSize, 0, Math.PI * 2);
        this.ctx.fill();
      }
    });

    this.canvas.addEventListener("wheel", (e: WheelEvent) => {
      if (e.deltaY > 0) {
        this.brushSize = Math.max(1, this.brushSize - 1);
      } else {
        this.brushSize += 1;
      }
      this.cursor.style.width = this.brushSize * 2 + "px";
      this.cursor.style.height = this.brushSize * 2 + "px";
    });
  }

  clear() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }
  setDrawingEnabled(enabled: boolean) {
    if (enabled) {
      this.canvas.addEventListener("mousedown", this.onMouseDown);
      document.body.addEventListener("mouseup", this.onMouseUp);
    } else {
      this.canvas.removeEventListener("mousedown", this.onMouseDown);
      document.body.removeEventListener("mouseup", this.onMouseUp);
    }
  }

  onMouseDown = (e: MouseEvent) => {
    this.drawing = true;
  };
  onMouseUp = (e: MouseEvent) => {
    this.drawing = false;
    this.history.add(this.canvas.toDataURL());
  };
}
