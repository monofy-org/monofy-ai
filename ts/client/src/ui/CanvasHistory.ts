export class CanvasHistory {
  ctx: CanvasRenderingContext2D;
  history: string[];
  currentIndex: number = 0;
  constructor(ctx: CanvasRenderingContext2D) {
    this.ctx = ctx;
    this.history = [];
  }
  add(base64_image: string) {
    this.history.push(base64_image);
  }
  undo() {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      return this.history[this.currentIndex];
    }
    return null;
  }
  redo() {
    if (this.currentIndex < this.history.length - 1) {
      this.currentIndex++;
      return this.history[this.currentIndex];
    }
    return null;
  }
}
