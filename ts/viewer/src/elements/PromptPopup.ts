export class PromptPopup {
  public readonly domElement: HTMLElement;
  private readonly _input: HTMLInputElement;
  private readonly _button: HTMLButtonElement;
  private readonly _header: HTMLElement;
  private readonly _text: HTMLParagraphElement;

  constructor(
    container: HTMLElement,
    private readonly onOk: (value: string) => void
  ) {
    this.domElement = document.createElement("div");
    this.domElement.classList.add("prompt-popup");
    this._header = document.createElement("h1");
    this.domElement.appendChild(this._header);

    this._text = document.createElement("p");
    this.domElement.appendChild(this._text);

    this._input = document.createElement("input");
    this._input.classList.add("prompt-input");
    this.domElement.appendChild(this._input);

    this._input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        this.onOk(this.getValue());
        this.hide();
      }
    });

    this._button = document.createElement("button");
    this._button.classList.add("prompt-button");
    this._button.textContent = "OK";
    this._button.addEventListener("click", () => {
      this.onOk(this.getValue());
      this.hide();
    });
    this.domElement.appendChild(this._button);

    container.appendChild(this.domElement);

    this.hide();
  }

  public isVisible() {
    return this.domElement.style.display !== "none";
  }

  public show(
    title: string,
    text: string = "",
    position?: { x: number; y: number }
  ) {
    this._header.textContent = title;
    this._text.textContent = text;
    this.domElement.style.display = "block";
    if (position) {
      this.domElement.style.left = `${position.x}px`;
      this.domElement.style.top = `${position.y}px`;
    }
    this._input.focus();
  }

  public hide() {
    this.domElement.style.display = "none";
  }

  public getValue() {
    return this._input.value;
  }
}
