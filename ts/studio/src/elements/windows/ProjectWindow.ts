import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { ContextMenu } from "../../../../elements/src/elements/ContextMenu";
import { AudioClock } from "../components/AudioClock";
import { ProjectUI } from "../ProjectUI";
import { IPattern } from "../../schema";
import { Playlist } from "../components/Playlist";
import { ProjectTreeView } from "../ProjectTreeView";
import { FileImporter } from "../../../../elements/src/importers/FileImporter";
import { ImagePreviewWindow } from "./ImagePreviewWindow";
import { GraphicsHelpers } from "../../abstracts/GraphicsHelpers";
import { WaveEditorWindow } from "./WaveEditorWindow";
import { TreeViewItem } from "../../../../elements/src/elements/TreeView";

export class ProjectWindow extends DraggableWindow {
  readonly playlist: Playlist;

  beatWidth = 10;
  treeView: ProjectTreeView;

  get audioClock(): AudioClock {
    return this.ui.project.audioClock;
  }

  constructor(readonly ui: ProjectUI) {
    const container = document.createElement("div");
    container.classList.add("project-container");

    super(ui.container, {
      title: "Project Timeline",
      persistent: true,
      content: container,
      width: 800,
      height: 400,
    });

    this.playlist = new Playlist(this.ui.project);
    this.playlist.grid.drawingEnabled = false;

    const sourceContainer = document.createElement("div");
    sourceContainer.classList.add("project-source-container");

    this.treeView = new ProjectTreeView(this.ui);
    this.treeView.on("select", () => {
      const item = this.treeView.selectedItem;
      if (!item) throw new Error("No item selected");

      console.log("ProjectTreeView.select", item);

      if (item.type === "pattern") {
        const pattern = item.data as IPattern;

        //if (!pattern.image) {
        const canvas = document.createElement("canvas");
        canvas.width = 1024;
        canvas.height = 100;
        GraphicsHelpers.renderPattern(canvas, pattern, "#bbaaff", 25);
        pattern.image = canvas.toDataURL();
        //}

        this.playlist.grid.drawingLabel = pattern.name;
        this.playlist.grid.drawingValue = this.treeView.selectedItem;
        this.playlist.grid.drawingEnabled = true;
        this.playlist.grid.drawingImage = pattern.image;
      } else if (item.type === "image") {
        const file = item.data as File;
        this.playlist.grid.drawingImage = URL.createObjectURL(file);
        this.playlist.grid.drawingLabel = file.name;
        this.playlist.grid.drawingValue = this.treeView.selectedItem;
        this.playlist.grid.drawingEnabled = true;
      } else if (item.type === "audio") {
        const audioItem = item.data as {
          buffer: AudioBuffer;
          image: string;
        };
        this.playlist.grid.drawingImage = audioItem.image;
        this.playlist.grid.drawingLabel = item.name;
        this.playlist.grid.drawingValue = item;
        this.playlist.grid.drawingEnabled = true;
      } else {
        this.playlist.grid.drawingEnabled = false;
      }
    });

    this.treeView.on("open", (item) => {
      if (!(item instanceof TreeViewItem)) return;

      console.log("ProjectTreeView.dblclick", item);
      if (item.type === "audio") {
        const data = item.data as {
          buffer: AudioBuffer;
          image: string;
          name: string;
        };
        const editor = new WaveEditorWindow(ui, data).show();
        editor.on("change", () => {
          console.log("WaveEditorWindow.change", editor.waveEditor.audioBuffer);
          if (!editor.waveEditor.audioBuffer)
            throw new Error("No audio buffer");
          data.buffer = editor.waveEditor.audioBuffer;
        });
      } else if (item.type === "image") {
        const file = item.data as File;
        new ImagePreviewWindow(ui, file).show();
      } else if (item.type === "pattern") {
        const pattern = item.data as IPattern;
        this.ui.patternWindow.loadPattern(pattern);
        this.ui.patternWindow.show();
      }
    });

    this.treeView.on("change", (e) => {
      console.log("ProjectTreeView.change", e);
    });

    this.on("keydown", (e) => {
      const event = e as KeyboardEvent;
      console.log("ProjectWindow.keydown", event.key, event.ctrlKey);      
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
