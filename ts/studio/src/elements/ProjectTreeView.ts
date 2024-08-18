import { ContextMenu } from "../../../elements/src/elements/ContextMenu";
import {
  TreeView,
  TreeViewFolder,
} from "../../../elements/src/elements/TreeView";
import { GraphicsHelpers } from "../abstracts/GraphicsHelpers";
import { AudioImporter } from "../../../elements/src/importers/AudioImporter";
import { FileImporter } from "../../../elements/src/importers/FileImporter";
import { IPattern } from "../schema";
import { IProjectUpdateEvent } from "./Project";
import type { ProjectUI } from "./ProjectUI";
import JSZip from "jszip";
import { getAudioContext } from "../../../elements/src/managers/AudioManager";
import { parseMidiFileToEventsByChannel } from "../importers/MidiImporter";

export class ProjectTreeView extends TreeView {
  loadFile(file: File) {
    if (file.type.startsWith("audio/")) {
      this.loadAudioFile(file);
    } else if (file.type.startsWith("image/")) {
      this.loadImageFile(file);
    } else if (file.type.startsWith("video/")) {
      this.loadVideoFile(file);
    } else {
      throw new Error("Unsupported file type");
    }
  }

  constructor(readonly ui: ProjectUI) {
    super("Project");

    const folderMenu = new ContextMenu(document.body, this.domElement, () => {
      return (
        this.selectedItem?.type === "folder" && this.selectedItem.id !== "root"
      );
    });

    folderMenu.addItem("New Folder", () => {
      this.selectedItem = this.newFolderInSelected();
    });

    folderMenu.addItem(
      "Rename",
      () => {
        this.selectedItem?.rename();
      },
      () => {
        return this.selectedItem?.parentId !== this.root.id;
      }
    );

    folderMenu.addItem(
      "Import Audio",
      () => {
        folderMenu.hide();
        FileImporter.importFile("audio/*").then((file) => {
          const folder = this.selectedItem! as TreeViewFolder;
          this.loadAudioFile(file, folder);
        });
      },
      () => {
        return this.selectedItem?.id === "audio";
      }
    );

    folderMenu.addItem(
      "Import Stems",
      () => {
        folderMenu.hide();
        FileImporter.importFile("application/zip").then((file) => {
          const folder = this.selectedItem! as TreeViewFolder;
          this.loadStemsFromZip(file, folder);
        });
      },
      () => {
        return this.selectedItem?.id === "audio";
      }
    );

    folderMenu.addItem("Import MIDI File", () => {
      folderMenu.hide();
      FileImporter.importFile("audio/midi").then((file) => {
        parseMidiFileToEventsByChannel(file, ui.project.audioClock.bpm).then(
          (pattern) => {
            const folder = this.selectedItem! as TreeViewFolder;
            this.selectedItem = folder.add(
              "pattern",
              file.name,
              undefined,
              pattern
            );
            this.emit("open", this.selectedItem);
          }
        );
      });
    });

    folderMenu.addItem(
      "Import Video",
      () => {
        folderMenu.hide();
        FileImporter.importFile("video/*").then((file) => {
          console.log("Imported file", file);
        });
      },
      () => {
        return this.selectedItem?.id === "video";
      }
    );

    folderMenu.addItem(
      "Import Image",
      () => {
        folderMenu.hide();
        FileImporter.importFile("image/*").then((file) => {
          const folder = this.selectedItem! as TreeViewFolder;
          this.selectedItem = folder.add("image", file.name, undefined, file);
          this.emit("open", this.selectedItem);
        });
      },
      () => {
        return this.selectedItem?.id === "images";
      }
    );

    folderMenu.addItem(
      "New Pattern",
      () => {
        const folder = this.selectedItem! as TreeViewFolder;
        const pattern: IPattern = {
          name: "New Pattern",
          image: "",
          tracks: [],
        };
        this.ui.project.patterns.push(pattern);
        this.selectedItem = folder.add(
          "pattern",
          pattern.name,
          undefined,
          pattern
        );
      },
      () => {
        return this.selectedItem?.id === "patterns";
      }
    );

    const fileMenu = new ContextMenu(document.body, this.domElement, () => {
      return this.selectedItem?.type !== "folder";
    });

    fileMenu.addItem(
      "Open",
      () => {
        if (this.selectedItem) this.emit("open", this.selectedItem);
      },
      () => {
        return Boolean(this.selectedItem);
      }
    );

    fileMenu.addItem(
      "Duplicate",
      () => {
        this.selectedItem = this.duplicateSelected();
      },
      () => {
        return Boolean(this.selectedItem);
      }
    );

    fileMenu.addItem(
      "Rename",
      () => {
        this.selectedItem?.rename();
      },
      () => {
        return Boolean(this.selectedItem);
      }
    );

    ui.project.on("update", (e) => {
      const update = e as IProjectUpdateEvent;
      if (update.type === "project") {
        console.log("ProjectTreeView.update");
        this.clear();
        this.root.add("folder", "Audio", "audio");
        this.root.add("folder", "Images", "images");
        const patternsFolder: TreeViewFolder = this.root.add(
          "folder",
          "Patterns",
          "patterns"
        );
        patternsFolder.toggle(true);
        const presetsFolder: TreeViewFolder = this.root.add(
          "folder",
          "Presets",
          "presets"
        );
        presetsFolder.toggle(true);
        presetsFolder.add("folder", "FM Bass");
        presetsFolder.add("folder", "FM Piano");
        presetsFolder.add("folder", "Mixer");
        presetsFolder.add("folder", "Multisampler");
        this.root.add("folder", "Samples", "samples");
        this.root.add("folder", "Video", "video");
        for (const pattern of ui.project.patterns) {
          patternsFolder.add("pattern", pattern.name, undefined, pattern);
        }
      }
    });
  }

  loadAudioFile(file: File, folder: TreeViewFolder = this.get("audio")!) {
    AudioImporter.loadFile(file, this.ui.audioContext)
      .then((buffer) => {
        if (!buffer) {
          throw new Error("Error loading audio file");
        }

        const canvas = document.createElement("canvas");
        canvas.width = 2048;
        canvas.height = 100;

        GraphicsHelpers.renderWaveform(canvas, buffer, "#bbaaffaa");

        this.selectedItem = folder.add("audio", file.name, undefined, {
          buffer,
          image: canvas.toDataURL(),
          name: file.name,
        });

        this.emit("open", this.selectedItem);
      })
      .catch((error) => {
        console.error("Error loading audio file", error);
        alert("Error loading audio file");
      });
  }

  loadStemsFromZip(
    zipfile: File,
    folder: TreeViewFolder = this.get("samples")!
  ) {
    // unzip using jszip
    const zip = new JSZip();
    zip.loadAsync(zipfile).then(async (zip) => {
      const entries = Object.keys(zip.files);
      for (const entry of entries) {
        const file = zip.files[entry];
        if (file.dir) continue;
        const s = file.name.split(".");
        const name = s[0];
        const ext = s[1];
        // decode audio buffer
        if (ext !== "wav" && ext !== "mp3") {
          alert("Skipping file: " + file.name);
          continue;
        }
        const data = await file.async("arraybuffer");
        const audioContext = getAudioContext();
        await audioContext.decodeAudioData(data).then((buffer) => {
          const stem = {
            buffer,
            name,
          };
          this.selectedItem = folder.add("audio", name, undefined, stem);
        });
      }
    });
  }

  loadImageFile(file: File, folder: TreeViewFolder = this.get("images")!) {
    this.selectedItem = folder.add("image", file.name, undefined, file);
    this.emit("open", this.selectedItem);
  }

  loadVideoFile(file: File, folder: TreeViewFolder = this.get("video")!) {
    const reader = new FileReader();
    reader.onload = () => {
      const url = reader.result as string;
      this.selectedItem = folder.add("video", file.name, undefined, url);
    };

    reader.readAsDataURL(file);
  }
}
