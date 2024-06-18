export class ContextMenu {
  private readonly domElement: HTMLElement;
  private readonly _submenus: ContextMenu[] = [];

  constructor(
    private readonly container?: HTMLElement,
    private readonly attachedElement?: HTMLElement
  ) {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("context-menu");

    if (container) container.appendChild(this.domElement);

    if (attachedElement) {
      this.hide();
      attachedElement.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        this.show(event.clientX + 10, event.clientY - 5);
      });
    }
    
  }

  private _handlePointerDown(e: MouseEvent) {
    if (
      e.target !== this.domElement &&
      !this.domElement.contains(e.target as Node)
    )
      this.hide();
  }

  public isVisible() {
    return this.domElement.style.display !== "none";
  }

  public addItem(label: string, callback: () => void) {
    const item = document.createElement("div");
    item.classList.add("context-menu-item");
    item.textContent = label;
    item.addEventListener("click", () => {
      callback();
      this.hide();
    });
    this.domElement.appendChild(item);
  }

  public addSubmenu(label: string, submenu: ContextMenu) {
    const item = document.createElement("div");
    item.classList.add("context-menu-item");
    item.textContent = label;
    item.addEventListener("click", (e) => {
      submenu.show(
        this.domElement.offsetLeft + this.domElement.offsetWidth - 5,
        item.offsetTop
      );
      const elt = e.target as HTMLElement;
      if (elt.classList.contains("context-menu-item")) {
        this.hide();
      }
    });

    submenu.domElement.classList.add("context-menu-submenu");
    this._submenus.push(submenu);

    item.appendChild(submenu.domElement);
    this.domElement.appendChild(item);
  }

  public show(x: number, y: number) {
    if (!this.container) return;
    this.domElement.style.left = `${x}px`;
    this.domElement.style.top = `${y}px`;
    this.domElement.style.display = "block";
    document.addEventListener("pointerdown", (e) => this._handlePointerDown(e));
  }

  public hide() {
    if (!this.isVisible() || !this.container) return;
    this.domElement.style.display = "none";
    for (let i = 0; i < this._submenus.length; i++) {
      this._submenus[i].hide();      
    }
    document.removeEventListener("pointerdown", (e) => this._handlePointerDown(e));
  }
}
