import { SelectableElement } from "../../../elements/src/elements/SelectableElement";
import { SelectableGroup } from "../../../elements/src/elements/SelectableGroup";
import { IPattern } from "../schema";

export class PlaylistSourceItem extends SelectableElement {
  constructor(group: SelectableGroup<PlaylistSourceItem>, readonly item: IPattern) {
    super(group, "div", "playlist-source-item");

    this.domElement.innerText = item.name;
  }
}
