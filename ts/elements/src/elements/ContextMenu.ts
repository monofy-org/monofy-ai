import EventObject from "../EventObject";

export class ContextMenu extends EventObject<"shown"> {
  private readonly domElement: HTMLElement;
  private readonly _submenus: ContextMenu[] = [];

  constructor(    
    private readonly attachedElement?: HTMLElement,
    filter?: (e?: PointerEvent) => boolean
  ) {
    super();

    this.domElement = document.createElement("div");
    this.domElement.classList.add("context-menu");

    document.body.appendChild(this.domElement);

    if (this.attachedElement) {
      this.hide();
      this.attachedElement.addEventListener("contextmenu", (event) => {
        if (!event.ctrlKey) {
          if (filter && !filter()) return;
          event.preventDefault();
          this.show(event.clientX + 10, event.clientY - 5, event);
        }
      });
    }
  }

  private _handlePointerDown() {
    // only hide the main menu (others will follow using css rules)
    if (!this.domElement.classList.contains("submenu")) this.hide();
  }

  public isVisible() {
    return this.domElement.style.display !== "none";
  }

  public addItem(label: string, callback: (e: PointerEvent) => void, filter?: (e?: PointerEvent) => boolean) {
    const item = document.createElement("div");
    item.classList.add("context-menu-item");
    item.textContent = label;

    let evt: PointerEvent | undefined = undefined;
    item.addEventListener("pointerdown", (e) => {      
      callback(evt!);
    });
    this.domElement.appendChild(item);

    this.on("shown", (e: any) => {
      evt = e;
      if (filter && !filter(e as PointerEvent)) item.style.display = "none";
      else item.style.display = "block";
    });

    return item;
  }

  public addSubmenu(label: string, submenu: ContextMenu) {
    const item = document.createElement("div");
    item.classList.add("context-menu-item");
    item.classList.add("submenu");
    item.textContent = label;
    item.addEventListener("pointerdown", (e: PointerEvent) => {
      submenu.show(
        this.domElement.offsetLeft + this.domElement.offsetWidth - 5,
        item.offsetTop,
        e
      );
      // const elt = e.target as HTMLElement;
      // if (elt.classList.contains("context-menu-item")) {
      //   this.hide();
      // }
    });

    submenu.domElement.classList.add("context-menu-submenu");
    this._submenus.push(submenu);

    item.appendChild(submenu.domElement);
    this.domElement.appendChild(item);
  }

  public show(x: number, y: number, event?: Event) {
    this.domElement.style.left = `${x}px`;
    this.domElement.style.top = `${y}px`;
    this.domElement.style.display = "flex";
    document.addEventListener("pointerup", () => this._handlePointerDown());
    this.emit("shown", event);
  }

  public hide() {
    if (!this.isVisible()) return;
    this.domElement.style.display = "none";
    // for (let i = 0; i < this._submenus.length; i++) {
    //   this._submenus[i].hide();
    // }
    document.removeEventListener("pointerup", () => this._handlePointerDown());
  }
}
