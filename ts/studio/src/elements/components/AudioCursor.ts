import { ICursorTimeline } from "../ICursorTimeline";

export class AudioCursor {
  readonly domElement: HTMLDivElement;

  constructor(    
    readonly timeline: ICursorTimeline
  ) {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("audio-cursor");

    const clock = timeline.audioClock;

    clock.on("update", () => {
      this.update();
    });


    clock.on("start", () => {
      this.domElement.style.transform = "translateX(0)";
      this.domElement.style.display = "block";
      this.domElement.parentElement?.appendChild(this.domElement);      
    });


    clock.on("stop", () => {
      this.domElement.style.display = "none";
    });
  }

  update() {
    this.domElement.style.transform = `translateX(${
      this.timeline.audioClock.currentBeat * 100
    }px)`;
    console.log("update");
  }

  hide() {
    this.domElement.style.display = "none";
  }
}
