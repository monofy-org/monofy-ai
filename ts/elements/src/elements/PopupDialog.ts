import { BaseElement } from "./BaseElement";

export class PopupDialog extends BaseElement<"result" | "cancel"> {  
  public readonly closeButton: HTMLButtonElement;
  public readonly okButton: HTMLButtonElement | undefined;
  public readonly cancelButton: HTMLButtonElement | undefined;
  public readonly content: HTMLDivElement;

  private _content: HTMLElement | undefined;
  private _result: string | undefined;

  get isVisibile() {
    return this.domElement.style.display !== "none";
  }

  constructor(
    content: HTMLElement,
    ok: string | HTMLButtonElement = "OK",
    cancel: string | HTMLButtonElement = "Cancel", 
  ) {
    super("div", "dialog-popup");

    console.assert(Boolean(ok && cancel), "ok and cancel must be defined");

    this.domElement.style.display = "none";
    
    this._content = content.cloneNode(true) as HTMLDivElement;
    this.content = document.createElement("div");
    this.content.classList.add("dialog-popup-content");
    this.domElement.appendChild(this.content);    
    this.content.appendChild(content);  

    this.closeButton = document.createElement("button");
    this.closeButton.classList.add("window-close-button");    
    this.domElement.appendChild(this.closeButton);
    this.closeButton.addEventListener("pointerdown", () => {
      this.domElement.style.display = "none";
    });

    if (ok instanceof HTMLButtonElement) {
      this.okButton = ok;
    } else if (ok) {
      this.okButton = document.createElement("button");
      this.okButton.textContent = ok;
      this.content.appendChild(this.okButton);
    }

    if (this.okButton) {
      this.okButton.addEventListener("click", () => {
        this.emit("result", this);
      });
    }

    if (cancel instanceof HTMLButtonElement) {
      this.cancelButton = cancel;
    } else if (cancel) {
      this.cancelButton = document.createElement("button");
      this.cancelButton.textContent = cancel;
      this.content.appendChild(this.cancelButton);
    }

    if (this.cancelButton) {
      this.cancelButton.addEventListener("click", () => {
        this.emit("cancel", this);
        this.hide();   
      });
    }
  }

  show(x: number, y: number, targetObject?: unknown) { /* eslint-disable-line @typescript-eslint/no-unused-vars */
    this.domElement.style.display = "block";
    this.domElement.style.left = `${x}px`;
    this.domElement.style.top = `${y}px`;
    this.domElement.parentElement?.appendChild(this.domElement);
  }

  hide() {
    this.domElement.style.display = "none";
  }
}
