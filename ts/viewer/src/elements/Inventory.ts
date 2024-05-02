import { Storage, StorageItem } from "../../../elements/src/elements/Storage";

export class Inventory {
  public readonly domElement: HTMLElement;
  private readonly _fileList: HTMLElement;
  private readonly _storage = new Storage("inventory");

  constructor() {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("inventory");

    const header = document.createElement("h1");
    header.textContent = "Inventory";
    this.domElement.appendChild(header);

    this._fileList = document.createElement("ul");
    this.domElement.appendChild(this._fileList);

    const storage = new Storage("inventory");

    console.log("Storage", storage);

    storage.getFolderItems(null).then((items) => {
      console.log("Items", items);
      items.forEach((item) => {
        if (item instanceof InventoryItem) {
          this.add(item);
        } else {
          console.error("Error", "Unexpected item type in inventory");
        }
      });
    });
  }

  public add(item: InventoryItem) {
    const li = document.createElement("li");
    li.textContent = item.name;
    this._fileList.appendChild(li);
    li.setAttribute("id", item.id);
  }

  create(item: InventoryItem) {
    this.add(item);
    this._storage.add(item);
  }

  public remove(id: string) {
    const li = this._fileList.querySelector(`#${id}`);
    if (li) {
      this._fileList.removeChild(li);
      this._storage.delete(id);
    }
  }
}

export class InventoryItem extends StorageItem {
  constructor(
    name: string,
    data?: InventoryItem,
    public readonly parentId: string | null = null
  ) {
    super(name, data, parentId);
  }
}
