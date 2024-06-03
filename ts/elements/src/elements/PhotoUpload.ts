import { BaseElement } from "./BaseElement";

export class PhotoUpload extends BaseElement<"update"> {
  constructor() {
    super("div", "photo-upload");
    this.domElement.addEventListener("click", () => {
      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = "image/*";
      fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = () => {
            this.domElement.style.backgroundImage = `url(${reader.result})`;
            this.emit("update", reader.result);
          };
          reader.readAsDataURL(file);
        }
      });
      fileInput.click();
    });    
  }
}