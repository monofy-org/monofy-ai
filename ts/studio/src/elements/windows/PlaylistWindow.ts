import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { Composition } from "../Composition";
import { PlaylistTrack } from "../components/PlaylistTrack";
import { AudioClock } from "../components/AudioClock";

export class PlaylistWindow extends DraggableWindow<"update" | "select"> {
  private _trackContainer: HTMLDivElement;
  private _tracks: PlaylistTrack[] = [];
  private cursor: HTMLDivElement;

  constructor(
    readonly clock: AudioClock,
    readonly composition: Composition
  ) {
    const container = document.createElement("div");
    container.classList.add("playlist-track-container");

    super("Playlist", true, container);
    this._trackContainer = container;
    this.setSize(800, 400);

    this.cursor = document.createElement("div");
    this.cursor.classList.add("piano-roll-cursor");
    this.cursor.style.display = "none";
    this._trackContainer.appendChild(this.cursor);

    clock.on("update", () => {
      this.updateCursor();
    });
  }

  addTrack(name: string) {
    const track = new PlaylistTrack(name);
    this._tracks.push(track);
    this._trackContainer.appendChild(track.domElement);
    this._trackContainer.appendChild(this.cursor);
  }

  updateCursor() {
    if (this.clock.isPlaying) {
      this.cursor.style.transform = `translateX(${
        (this.clock.currentBeat * this._trackContainer.offsetWidth) / 16
      }px)`;
    }
  }
}
