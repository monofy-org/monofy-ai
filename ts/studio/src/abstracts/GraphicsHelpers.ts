import { IEvent, IPattern } from "../schema";

export abstract class GraphicsHelpers {
  static renderSequence(
    canvas: HTMLCanvasElement,
    events: IEvent[],
    color: string,
    beatWidth: number,
    clearFirst = true
  ) {
    // find lowest note
    let lowestNote = 88;
    let highestNote = 0;
    for (const event of events) {
      if (!(event.note || event.note === 0)) {
        console.log("event", event);
        throw new Error("renderToCanvas() No note!");
      }
      if (event.note < lowestNote) lowestNote = event.note;
      if (event.note > highestNote) highestNote = event.note;
    }

    highestNote += 12;
    lowestNote -= 12;

    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

    if (clearFirst) ctx.clearRect(0, 0, canvas.width, canvas.height);

    const paddedScale = Math.min(
      canvas.height / (highestNote - lowestNote + 1),
      2
    );

    for (const note of events) {
      const y = canvas.height - (note.note! - lowestNote + 1) * paddedScale;
      const x = note.start * beatWidth;
      const width = note.duration * beatWidth;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, width, paddedScale);
    }
  }

  static renderPattern(
    canvas: HTMLCanvasElement,
    pattern: IPattern,
    color: string,
    beatWidth: number
  ) {
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const concat: IEvent[] = [];

    for (const sequence of pattern.tracks) {
      concat.push(...sequence.events);
    }

    GraphicsHelpers.renderSequence(canvas, concat, color, beatWidth, false);
  }
}
