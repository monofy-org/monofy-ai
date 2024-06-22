import { BaseElement } from "./BaseElement";

export class PopupNotification extends BaseElement<"close"> {
  private readonly _closeButton: HTMLButtonElement;

  constructor(title: string, options?: NotificationOptions | undefined) {
    super("div", "notification");

    this.domElement.style.display = "hidden";

    const titleElt = document.createElement("h1");
    const messageElt = document.createElement("p");

    this._closeButton = document.createElement("button");
    this._closeButton.textContent = "×";
    const onclick = () => {
      this._closeButton.removeEventListener("click", onclick);
      this.dispose();
      this.emit("close");
    };
    this._closeButton.addEventListener("click", onclick);

    if (options?.body) {
      messageElt.textContent = options.body;
    }

    titleElt.textContent = title;

    this.domElement.appendChild(titleElt);
    this.domElement.appendChild(messageElt);
    this.domElement.appendChild(this._closeButton);

    document.body.appendChild(this.domElement);
  }
}
