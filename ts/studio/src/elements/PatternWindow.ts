import { DraggableWindow } from "../../../elements/src/elements/DraggableWindow";
import { Composition } from "./Composition";
import { IGridItem } from "./Grid";
import { PatternTrack } from "./PatternTrack";

export class PatternWindow extends DraggableWindow<"select"> {
  private _trackContainer: HTMLDivElement;
  readonly tracks: PatternTrack[] = [];

  constructor(readonly composition: Composition) {
    const container = document.createElement("div");
    container.classList.add("pattern-track-container");

    super("Pattern", true, container);
    this._trackContainer = container;
    this.setSize(800, 400);
    this.show(100, 100);
  }

  addTrack(name: string, pattern: IGridItem[] = []) {
    const track = new PatternTrack(name, pattern);
    track.on("select", (selectedTrack) => {
      this.fireEvent("select", selectedTrack);
    });
    this.tracks.push(track);
    this._trackContainer.appendChild(track.domElement);
  }

  removeTrack(track: PatternTrack) {
    const index = this.tracks.indexOf(track);
    if (index !== -1) {
      this.tracks.splice(index, 1);
      this._trackContainer.removeChild(track.domElement);
    }
  }
}
