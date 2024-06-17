import EventObject from "../../../../elements/src/EventObject";
import { IProject } from "../../schema";
import { Project } from "../Project";
import { AudioClock } from "./AudioClock";
import { AudioCursor, ICursorTimeline } from "./AudioCursor";
import { Grid } from "./Grid";
import { PlaylistTrack } from "./PlaylistTrack";

export class Playlist extends EventObject<"update"> implements ICursorTimeline {
  readonly domElement: HTMLDivElement;
  readonly grid: Grid;
  readonly _trackPanels: HTMLDivElement;
  readonly cursor: AudioCursor;
  readonly timeline: HTMLElement;
  beatWidth = 100;

  get audioClock(): AudioClock {
    return this.project.audioClock;
  }

  constructor(readonly project: Project) {
    super();

    this.domElement = document.createElement("div");
    this.domElement.classList.add("playlist");

    this._trackPanels = document.createElement("div");
    this._trackPanels.classList.add("playlist-tracks");
    this.domElement.appendChild(this._trackPanels);

    this.grid = new Grid(16, 60);
    this.domElement.appendChild(this.grid.domElement);
    this.grid.linkElement(this._trackPanels);
    this.grid.load(this.project.playlist);

    this.timeline = this.grid.domElement;

    this.cursor = new AudioCursor(this);
    this.cursor.domElement.classList.add("audio-cursor");
    this.grid.domElement.appendChild(this.cursor.domElement);
    this.cursor.domElement.style.marginLeft =
      this._trackPanels.offsetWidth + "px";
    this.grid.scrollTop = 0;
  }

  loadProject(project: IProject) {
    let i = 0;

    for (const track of project.tracks) {
      const playlistTrack = new PlaylistTrack(track.name);
      this._trackPanels.appendChild(playlistTrack.domElement);
      i++;
    }

    for (; i < 16; i++) {
      const playlistTrack = new PlaylistTrack(`Track ${i + 1}`);
      this._trackPanels.appendChild(playlistTrack.domElement);
    }
  }
}
