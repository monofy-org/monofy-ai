import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { ImagePreview } from "../../../../elements/src/elements/ImagePreview";

export class ImagePreviewWindow extends DraggableWindow {
  imagePreview: ImagePreview;
  constructor(file: File) {
    super({ title: file.name });

    this.imagePreview = new ImagePreview(file);
    this.domElement.appendChild(this.imagePreview.domElement);
  }
}
