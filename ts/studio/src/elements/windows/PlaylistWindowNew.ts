import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { ProjectUI } from "../ProjectUI";
import { Playlist } from "../components/Playlist";

export class PlaylistWindowNew extends DraggableWindow {
  constructor(ui: ProjectUI) {

    const playlist = new Playlist(ui.project);

    super({
      title: "Playlist",
      persistent: true,
      content: playlist.domElement,
      width: 300,
      height: 400,
      left: 100,
      top: 100,
    });
  }
}
