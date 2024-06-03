import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { Project } from "../Project";
import { PlaylistTrack } from "../components/PlaylistTrack";
import { AudioClock } from "../components/AudioClock";
import { AudioCursor, ICursorTimeline } from "../components/AudioCursor";

export class PlaylistWindow
  extends DraggableWindow<"update" | "select">
  implements ICursorTimeline
{
  private _tracks: PlaylistTrack[] = [];
  readonly timeline: HTMLDivElement;
  readonly cursor: AudioCursor;
  beatWidth = 100;

  get audioClock(): AudioClock {
    return this.project.audioClock;
  }

  constructor(readonly project: Project) {
    const container = document.createElement("div");
    container.classList.add("playlist-track-container");

    super("Playlist", true, container);
    this.timeline = container;
    this.setSize(800, 400);

    this.cursor = new AudioCursor(this);

    this.project.audioClock.on("update", () => {
      this.cursor.update();
    });
  }

  addTrack(name: string) {
    const track = new PlaylistTrack(name);
    this._tracks.push(track);
    this.timeline.appendChild(track.domElement);
    this.timeline.appendChild(this.cursor.domElement);
  }
}
