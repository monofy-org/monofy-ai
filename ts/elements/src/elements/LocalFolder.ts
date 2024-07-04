import { BaseElement } from "./BaseElement";

export class LocalFolder extends BaseElement {
  constructor() {
    super("div", "local-folder");

    const folderButton = document.createElement("button");
    folderButton.textContent = "Add Folder";
    folderButton.addEventListener("click", async () => {
      try {
        // Prompt user to select a directory.
        const dirHandle = await window.showDirectoryPicker();

        // Call a function to read and display files.
        await this.readDirectory(dirHandle);
      } catch (err) {
        console.error(err);
      }
    });
  }

  async readDirectory(dirHandle: FileSystemDirectoryHandle) {
    for await (const entry of dirHandle.values()) {
      if (entry.kind === "directory") {
        const folder = document.createElement("div");
        folder.textContent = entry.name;
        folder.onclick = async () => {
          await this.readDirectory(entry);
        };
        this.domElement.appendChild(folder);
      } else {
        const file = document.createElement("div");
        file.textContent = entry.name;
        this.domElement.appendChild(file);
      }
    }
  }
}
