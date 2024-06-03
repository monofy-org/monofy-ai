import { BaseElement } from "../../elements/src/elements/BaseElement";

export class PhotoBooth extends BaseElement<"update"> {
  constructor() {
    super("div", "photo-booth");
  }
}

(window as any).PhotoBooth = PhotoBooth;
