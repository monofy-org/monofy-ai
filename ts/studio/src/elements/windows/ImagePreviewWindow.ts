import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { ImagePreview } from "../../../../elements/src/elements/ImagePreview";
import type { ProjectUI } from "../ProjectUI";

export class ImagePreviewWindow extends DraggableWindow {
  imagePreview: ImagePreview;
  constructor(ui: ProjectUI, file: File) {
    super(ui.container, { title: file.name });

    this.imagePreview = new ImagePreview(file);
    this.content.appendChild(this.imagePreview.domElement);
  }
}
