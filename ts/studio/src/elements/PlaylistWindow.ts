import { DraggableWindow } from "../../../elements/src/elements/DraggableWindow";
import { Composition } from "./Composition";
import { PlaylistTrack } from "./PlaylistTrack";

export class PlaylistWindow extends DraggableWindow<"update" | "select"> {
  private _trackContainer: HTMLDivElement;
  private _tracks: PlaylistTrack[] = [];

  constructor(readonly composition: Composition) {
    const container = document.createElement("div");
    container.classList.add("playlist-track-container");

    super("Playlist", true, container);
    this._trackContainer = container;
    this.setSize(800, 400);
    this.show(100, 100);
  }

  addTrack(name: string) {
    const track = new PlaylistTrack(name);
    this._tracks.push(track);
    this._trackContainer.appendChild(track.domElement);
  }
}
