import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { ContextMenu } from "../../../../elements/src/elements/ContextMenu";
import { AudioClock } from "../components/AudioClock";
import { ProjectUI } from "../ProjectUI";
import { IPattern } from "../../schema";
import { Playlist } from "../components/Playlist";
import { ProjectTreeView } from "../ProjectTreeView";
import { FileImporter } from "../../importers/FileImporter";
import { ImagePreviewWindow } from "./ImagePreviewWindow";

export class PlaylistWindow extends DraggableWindow {
  readonly playlist: Playlist;

  beatWidth = 10;
  treeView: ProjectTreeView;

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
    this.playlist.grid.drawingEnabled = false;

    const sourceContainer = document.createElement("div");
    sourceContainer.classList.add("playlist-source-container");

    this.treeView = new ProjectTreeView(this.ui);
    this.treeView.on("select", () => {
      console.log(
        "ProjectTreeView.select",
        this.treeView.selectedItem?.type,
        this.treeView.selectedItem?.name,
        this.treeView.selectedItem?.data
      );
      if (this.treeView.selectedItem?.type === "pattern") {
        const pattern = this.treeView.selectedItem.data as IPattern;
        this.playlist.grid.drawingLabel = pattern.name;
        this.playlist.grid.drawingImage = pattern.image;
        this.playlist.grid.drawingValue = pattern;
        this.playlist.grid.drawingEnabled = true;
        this.ui.patternWindow.loadPattern(pattern);
      } else if (this.treeView.selectedItem?.type === "image") {
        const file = this.treeView.selectedItem.data as File;
        this.playlist.grid.drawingImage = URL.createObjectURL(file);
        this.playlist.grid.drawingLabel = file.name;
        this.playlist.grid.drawingValue = undefined;
        this.playlist.grid.drawingEnabled = true;
        const window = new ImagePreviewWindow(file);
        window.show();
      } else if (this.treeView.selectedItem?.type === "audio") {
        const item = this.treeView.selectedItem.data as {
          buffer: AudioBuffer;
          image: string;
        };
        this.playlist.grid.drawingImage = item.image;
        this.playlist.grid.drawingLabel = this.treeView.selectedItem.name;
        this.playlist.grid.drawingValue = item.buffer;
        this.playlist.grid.drawingEnabled = true;
      } else {
        this.playlist.grid.drawingEnabled = false;
      }
    });

    this.treeView.on("change", (e) => {
      console.log("ProjectTreeView.change", e);
    });

    const addMenu = new ContextMenu(document.body);

    addMenu.addItem("Import Audio", () => {
      FileImporter.importFile("audio/*").then((file) => {
        console.log("Imported file", file);
      });
    });

    const generateMenu = new ContextMenu();
    generateMenu.addItem("Stable Audio", () => {
      console.log("Stable Audio");
    });
    generateMenu.addItem("MusicGen", () => {
      console.log("MusicGen");
    });

    addMenu.addSubmenu("Generate", generateMenu);

    sourceContainer.appendChild(this.treeView.domElement);

    container.appendChild(sourceContainer);
    container.appendChild(this.playlist.domElement);
  }
}
