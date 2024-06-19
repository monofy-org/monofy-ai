import EventObject from "../../../../elements/src/EventObject";
import { IEvent, IPlaylistEvent, IProject, ITrackOptions } from "../../schema";
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
  beatWidth = 20;
  readonly _items: Map<IEvent, IPlaylistEvent> = new Map();

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

    this.grid = new Grid(16, this.beatWidth, 60);
    this.grid.on("add", (e) => {
      const event = e as IEvent;
      const playlistEvent = e as IPlaylistEvent;
      if (!this.project.playlist.events.includes(playlistEvent)) {
        this.project.playlist.events.push(playlistEvent);
      }
      this._items.set(event, playlistEvent);
      this.emit("update");
      console.log("Added event", e);
    });

    this.grid.on("remove", (e) => {      
      const event = e as IEvent;
      if (this._items.has(event)) {
        const playlistEvent = e as IPlaylistEvent;
        const index = this.project.playlist.events.indexOf(playlistEvent);
        if (index >= 0) {
          this.project.playlist.events.splice(index, 1);
          console.log("Removed event", e);
        }
        this._items.delete(event);
      }
      this.emit("update");
    });

    this.grid.quantize = 1;
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
      const playlistTrack = new PlaylistTrack(track);
      this._trackPanels.appendChild(playlistTrack.domElement);
      i++;
    }

    for (; i < 16; i++) {
      const track: ITrackOptions = {
        name: `Track ${i + 1}`,
        mute: false,
        solo: false,
        selected: false,
      };
      const playlistTrack = new PlaylistTrack(track);
      project.tracks.push(track);
      this._trackPanels.appendChild(playlistTrack.domElement);
    }
  }
}
