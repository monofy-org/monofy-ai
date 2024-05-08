import { ICursorTimeline } from "../ICursorTimeline";
import { AudioClock } from "./AudioClock";

export class AudioCursor {
  readonly domElement: HTMLDivElement;

  constructor(
    readonly audioClock: AudioClock,
    readonly container: ICursorTimeline,
  ) {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("audio-cursor");

    audioClock.on("update", () => {
      this.update();
    });
  }

  update() {
    // Update the cursor position
  }
}
