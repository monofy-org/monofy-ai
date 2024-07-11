import { UUID } from "../UUID";
import { BaseElement } from "./BaseElement";

type TreeViewItemType = "file" | "folder" | string;

interface ITreeViewItem {
  type: TreeViewItemType;
  name: string;
  id: string;
  parentId: string | null;
  data?: unknown;
  thumbnail?: string;
}

export class TreeViewItem
  extends BaseElement<"add" | "change">
  implements ITreeViewItem
{
  protected _content: HTMLDivElement;
  protected _label: HTMLDivElement;
  protected _icon: HTMLDivElement;

  get name() {
    return this.item.name;
  }

  get id() {
    return this.item.id;
  }

  get parentId() {
    return this.item.parentId;
  }

  get type() {
    return this.item.type;
  }

  get data() {
    return this.item.data;
  }

  constructor(readonly item: ITreeViewItem) {
    super("div", "treeview-item");

    this._content = document.createElement("div");
    this._content.classList.add("treeview-item-content");

    this._icon = document.createElement("div");
    this._icon.classList.add("treeview-icon");
    this._content.appendChild(this._icon);

    this._label = document.createElement("div");
    this._label.textContent = item.name;
    this._label.classList.add("treeview-item-label");
    this._content.appendChild(this._label);
    this.domElement.id = item.id;
    this.domElement.classList.add(item.type);

    this.domElement.appendChild(this._content);
  }

  rename() {
    const newName = prompt("Enter new name", this.name);
    if (newName) {
      this.item.name = newName;
      const data = this.item.data as object;
      if (data && "name" in data) {
        data.name = newName;
      }
      this._label.textContent = newName;
      this.emit("change", this);
    }
  }
}

export class TreeViewFolder extends TreeViewItem {
  private readonly _expander: HTMLDivElement;
  private _childContainer?: HTMLDivElement;

  get expanded() {
    return this.domElement.classList.contains("expanded");
  }

  constructor(readonly item: ITreeViewItem) {
    super({ ...item, type: "folder" });

    this._expander = document.createElement("div");
    this._expander.classList.add("treeview-expander");
    this._expander.textContent = "+";
    this._content.insertBefore(this._expander, this._label);

    this._expander.addEventListener("pointerdown", () => this.toggle());

    this.domElement.addEventListener("dragover", (e) => {
      e.preventDefault();
    });

    this.domElement.addEventListener("drop", (e) => {
      e.preventDefault();
      const id = e.dataTransfer?.getData("text/plain");
      this.emit("change", { id, parentId: this.id });
    });

    this._content.addEventListener("dblclick", (e) => {
      if (e.target != this._content) return;
      this.toggle();
      e.preventDefault();
      e.stopPropagation();
    });
  }

  clear() {
    this._childContainer?.remove();
    this._childContainer = undefined;
  }

  toggle(expanded?: boolean) {
    this.domElement.classList.toggle("expanded", expanded);

    if (this.domElement.classList.contains("expanded")) {
      this._expander.textContent = "-";
    } else {
      this._expander.textContent = "+";
    }
  }

  add<T extends TreeViewItem>(
    type: TreeViewItemType,
    name: string,
    id?: string,
    data?: unknown
  ): T {
    const item = new (type == "folder" ? TreeViewFolder : TreeViewItem)({
      type,
      name,
      id: id || UUID.random(),
      parentId: this.id,
    });
    if (data) item.item.data = data;

    if (id) {
      console.log("Special item", id);
      item.domElement.classList.toggle("special", true);
    }

    if (!this._childContainer) {
      this._childContainer = document.createElement("div");
      this._childContainer.classList.add("treeview-children");
      this.domElement.appendChild(this._childContainer);
    }

    this._childContainer.appendChild(item.domElement);
    this.emit("add", item);
    item.on("add", (item) => this.emit("add", item));
    return item as T;
  }

  addItem(item: TreeViewItem) {
    if (item.domElement.contains(this.domElement)) {
      console.error("Cannot add parent to child");
      return;
    }

    if (!this._childContainer) {
      this._childContainer = document.createElement("div");
      this._childContainer.classList.add("treeview-children");
      this.domElement.appendChild(this._childContainer);
    }

    this._childContainer.appendChild(item.domElement);
  }
}

export class TreeView extends BaseElement<"select" | "change" | "open"> {
  readonly root: TreeViewFolder;
  private readonly _items: Map<string, TreeViewItem> = new Map();
  private _selectedItem: TreeViewItem | null = null;

  get selectedItem() {
    return this._selectedItem;
  }

  set selectedItem(value: TreeViewItem | null) {
    this._selectedItem?.domElement.classList.toggle("selected", false);
    if (value) {
      if (value.parentId) {
        const parent = this.get(value.parentId);
        if (parent instanceof TreeViewFolder) {
          parent.toggle(true);
        }
      }
      this._selectedItem = value;
      value.domElement.classList.toggle("selected", true);
      this.emit("select", value);
    } else {
      this._selectedItem = null;
    }
  }

  constructor(rootName = "/") {
    super("div", "treeview");
    this.root = new TreeViewFolder({
      type: "folder",
      name: rootName,
      id: "root",
      parentId: null,
    });
    this.root.domElement.classList.toggle("special", true);
    this._items.set(this.root.id, this.root);
    this.root.toggle(true);
    this.domElement.appendChild(this.root.domElement);

    this.root.on("add", (item) => {
      const newItem = item as TreeViewItem;
      this._items.set(newItem.id, newItem);
      newItem.on("change", (e) => {
        const evt = e as { id: string; parentId: string };
        const item = this._items.get(evt.id);
        const parent = this._items.get(evt.parentId);
        if (!item) {
          throw new Error("Item not found: " + evt.id);
        }
        if (!parent) {
          throw new Error("Parent not found: " + evt.parentId);
        }
        item.item.parentId = evt.parentId;
        parent.domElement.appendChild(item.domElement);
        this.emit("change", e);
      });
    });

    this.domElement.addEventListener("pointerdown", (e) => {
      const target = e.target as HTMLElement;

      if (!this.domElement.classList.contains("active")) {
        const onpointerdown = (p: PointerEvent) => {
          if (
            p.target &&
            !this.domElement.contains(p.target as HTMLElement) &&
            this.domElement.classList.contains("active")
          ) {
            this.domElement.classList.toggle("active", false);
            window.removeEventListener("pointerdown", onpointerdown);
            window.removeEventListener("keydown", onkeydown);
            console.log("Treeview inactive");
          }
        };

        const onkeydown = (e: Event) => {
          const event = e as KeyboardEvent;
          console.log("Keydown", event.key);

          if (
            event.key === "ArrowRight" &&
            this._selectedItem instanceof TreeViewFolder
          ) {
            this._selectedItem.toggle(true);
          } else if (
            event.key === "ArrowLeft" &&
            this._selectedItem instanceof TreeViewFolder
          ) {
            this._selectedItem.toggle(false);
          }
        };

        this.domElement.classList.toggle("active", true);
        window.addEventListener("pointerdown", onpointerdown);
        window.addEventListener("keydown", onkeydown);
        console.log("Treeview active");
      }

      if (target == this.domElement) {
        this.deselect();
        return;
      }

      if (!target.classList.contains("treeview-item-content")) {
        return;
      }

      const item = this._items.get(target.parentElement!.id);

      if (!item) {
        throw new Error("Item not found: " + target.parentElement!.id);
      }

      if (item != this._selectedItem) {
        this._selectedItem?.domElement.classList.toggle("selected", false);
        this._selectedItem = item;
        item.domElement.classList.toggle("selected", true);
        this.emit("select", item);
      }

      if (!item.domElement.classList.contains("special")) {
        console.log("TreeView: Drag started");
        this.domElement.classList.toggle("dragging", true);
      }
    });

    this.domElement.addEventListener("pointerup", (e) => {
      if (!this.domElement.classList.contains("dragging")) {
        return;
      }

      this.domElement.classList.toggle("dragging", false);

      console.log("Selected Element", this._selectedItem?.name);

      const elt = (e.target as HTMLElement)?.parentElement; // we are actually dropping it on _content

      if (this._selectedItem?.domElement.contains(elt)) {
        console.log("Dropped on self");
        return;
      }

      if (!elt) {
        console.error("No target element");
        return;
      }

      const target: TreeViewFolder = this.get(elt.id);

      if (!(target instanceof TreeViewFolder)) {
        console.error("Target is not a folder");
        return;
      }

      if (!target) {
        console.error("Target item not found");
        return;
      }

      if (this._selectedItem) {
        target.addItem(this._selectedItem);
        target.toggle(true);

        this.selectedItem = this._selectedItem;
      }
    });

    this.domElement.addEventListener("dblclick", (e) => {
      if (!this.selectedItem?.domElement.contains(e.target as HTMLElement))
        return;
      this.emit("open", this.selectedItem);
    });
  }

  get<T extends TreeViewItem>(id: string): T {
    return this._items.get(id) as T;
  }

  select(id: string) {
    const item = this._items.get(id);
    if (!item) return;
    this.selectedItem = item;
  }

  deselect() {
    this._selectedItem?.domElement.classList.toggle("selected", false);
    this._selectedItem = null;
  }

  clear() {
    this.root.clear();
    this._items.clear();
    this._items.set(this.root.id, this.root);
  }

  duplicateSelected() {
    if (!this.selectedItem) {
      throw new Error("No item selected");
    }

    const parentId = this.selectedItem?.parentId;
    if (!parentId) {
      throw new Error("Selected item has no parent");
    }

    const parent = this.get(parentId);

    if (!parent) {
      throw new Error("Parent not found");
    }
    if (parent instanceof TreeViewFolder) {
      const newItem = parent.add(
        this.selectedItem.type,
        this.selectedItem.name + " Copy",
        undefined,
        this.selectedItem.data
          ? JSON.parse(JSON.stringify(this.selectedItem.data))
          : undefined
      );
      const data = newItem.data as object;
      if ("name" in data) {
        data.name = newItem.name;
      }
      this.selectedItem = newItem;
      return newItem;
    }

    throw new Error("Selected item is not a folder");
  }

  newFolderInSelected() {
    let folder: TreeViewFolder;
    if (this.selectedItem) {
      if (this.selectedItem instanceof TreeViewFolder) {
        folder = this.selectedItem.add("folder", "New Folder");
        this.selectedItem.toggle(true);
      } else if (this.selectedItem.parentId) {
        const parent = this.get(this.selectedItem.parentId);
        if (parent && parent instanceof TreeViewFolder) {
          folder = parent.add("folder", "New Folder");
          parent.toggle(true);
          this.selectedItem = folder;
        } else {
          throw new Error("Parent is not a folder");
        }
      } else {
        throw new Error("Parent is not a folder");
      }
    } else {
      folder = this.root.add("folder", "New Folder");
    }
    folder.toggle(true);
    return folder;
  }
}
