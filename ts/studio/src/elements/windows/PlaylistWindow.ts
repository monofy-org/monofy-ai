import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { PlaylistTrack } from "../components/PlaylistTrack";
import { AudioClock } from "../components/AudioClock";
import { AudioCursor, ICursorTimeline } from "../components/AudioCursor";
import { ProjectUI } from "../ProjectUI";
import { IProject } from "../../schema";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import { PlaylistSourceItem } from "../PlaylistSourceItem";

export class PlaylistWindow extends DraggableWindow implements ICursorTimeline {
  private _tracks: PlaylistTrack[] = [];
  readonly timeline: HTMLDivElement;
  readonly bucket: SelectableGroup<PlaylistSourceItem>;
  readonly cursor: AudioCursor;
  beatWidth = 10;

  get audioClock(): AudioClock {
    return this.ui.project.audioClock;
  }

  constructor(readonly ui: ProjectUI) {
    const container = document.createElement("div");
    container.classList.add("playlist-container");

    super({
      title: "Playlist",
      persistent: true,
      content: container,
      width: 800,
      height: 400,
    });

    this.bucket = new SelectableGroup<PlaylistSourceItem>();
    this.bucket.domElement.classList.add("playlist-source-bucket");
    this.refresh();
    container.appendChild(this.bucket.domElement);

    this.timeline = document.createElement("div");
    this.timeline.classList.add("playlist-timeline");
    container.appendChild(this.timeline);

    this.cursor = new AudioCursor(this);   

    this.ui.project.audioClock.on("update", () => {
      this.cursor.update();
    });
  }

  refresh() {
    this.bucket.clear();

    for (const pattern of this.ui.project.patterns) {
      const item = new PlaylistSourceItem(this.bucket, pattern);
      this.bucket.addSelectable(item);
    }
  }

  addTrack(name: string) {
    console.log("Add track", name);
    const track = new PlaylistTrack(name);
    this._tracks.push(track);
    this.timeline.appendChild(track.domElement);
    this.timeline.appendChild(this.cursor.domElement);
  }

  loadProject(project: IProject) {
    this._tracks.forEach((track) => {
      this.timeline.removeChild(track.domElement);
    });
    this._tracks = [];

    project.timeline.forEach((track) => {
      this.addTrack(track.name);
    });

    for (let i = project.timeline.length; i < 16; i++) {
      this.addTrack(`Track ${i + 1}`);
    }
  }
}
