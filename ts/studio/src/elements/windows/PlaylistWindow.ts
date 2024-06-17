import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { AudioClock } from "../components/AudioClock";
import { ProjectUI } from "../ProjectUI";
import { IProject } from "../../schema";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import { PlaylistSourceItem } from "../PlaylistSourceItem";
import { Playlist } from "../components/Playlist";

export class PlaylistWindow extends DraggableWindow {  
  readonly bucket: SelectableGroup<PlaylistSourceItem>;
  readonly playlist: Playlist;    

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

    this.playlist = new Playlist(this.ui.project);

    this.bucket = new SelectableGroup<PlaylistSourceItem>();
    this.bucket.domElement.classList.add("playlist-source-bucket");
    this.refresh();

    container.appendChild(this.bucket.domElement);    
    container.appendChild(this.playlist.domElement);
  }

  refresh() {
    this.bucket.clear();

    for (const pattern of this.ui.project.patterns) {
      const item = new PlaylistSourceItem(this.bucket, pattern);
      this.bucket.addSelectable(item);
    }

    if (this.bucket.items.length > 0) {
      this.bucket.items[0].selected = true;
      this.playlist.grid.drawingEnabled = true;
      this.playlist.grid.drawingLabel = this.bucket.items[0].item.name;
      this.playlist.grid.drawingImage = this.bucket.items[0].item.name;
    } else {
      this.playlist.grid.drawingEnabled = false;
    }
  }

  loadProject(project: IProject) {
    this.playlist.loadProject(project);
  }
}
