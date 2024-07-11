import { TreeViewItem } from "../../../../elements/src/elements/TreeView";
import EventObject from "../../../../elements/src/EventObject";
import type { IAudioItem, IProject, ITrackOptions } from "../../schema";
import { IProjectUpdateEvent, Project } from "../Project";
import type { AudioClock } from "./AudioClock";
import { AudioCursor, ICursorTimeline } from "./AudioCursor";
import { Grid, GridItem } from "./Grid";
import { PlaylistTrack } from "./PlaylistTrack";

export class Playlist extends EventObject<"update"> implements ICursorTimeline {
  readonly domElement: HTMLDivElement;
  readonly grid: Grid;
  readonly _trackPanels: HTMLDivElement;
  readonly cursor: AudioCursor;
  readonly timeline: HTMLElement;
  beatWidth = 20;

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
      if (this.grid.drawingImage) {
        const gridItem = e as GridItem;
        const item = gridItem.value as TreeViewItem;
        if (!item) throw new Error("Invalid grid item");

        console.log("Playlist.add", item.type, item);

        if (item.type === "audio") {
          const audioItem = item.data as IAudioItem;
          if (audioItem) {
            const seconds = audioItem.buffer.duration;
            const beats = this.audioClock.getBeatTime(seconds, 0);            
            const backgroundSize = this.beatWidth * beats;

            console.log(
              "Added audio",
              audioItem.buffer.duration,
              backgroundSize
            );

            if (gridItem.image) {
              gridItem.image.style.backgroundSize = `${backgroundSize}px 100%`;
            }
          }
        }
      }
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

    this.project.on("update", (e) => {
      const update = e as IProjectUpdateEvent;
      console.log("ProjectUI project update", update);

      if (update.type === "project") {
        if (!(update.value instanceof Project)) {
          throw new Error("Invalid project update");
        }
        console.log("Playlist loaded new Project", update.value);
        this._loadProject(update.value);
      }
    });
  }

  private _loadProject(project: IProject) {
    let i = 0;

    for (let i = 0; i < project.playlist.tracks.length; i++) {
      const track = project.playlist.tracks[i];
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
      project.playlist.tracks.push(track);
      this._trackPanels.appendChild(playlistTrack.domElement);
    }

    console.log("Updating grid with new project playlist", project.playlist);

    this.grid.load(project.playlist);
  }
}
