import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { GraphicsHelpers } from "../../abstracts/GraphicsHelpers";

export class WaveEditor extends BaseElement<"update"> {
  _canvas: HTMLCanvasElement;
  private _nameElement: HTMLDivElement;

  constructor(name?: string, audioBuffer?: AudioBuffer) {
    super("div", "wave-editor");

    this._canvas = document.createElement("canvas");

    this._nameElement = document.createElement("div");
    this._nameElement.textContent = name || "Wave Editor";

    const toolsContainer = document.createElement("div");
    toolsContainer.classList.add("wave-editor-tools");

    this.domElement.appendChild(this._nameElement);
    this.domElement.appendChild(this._canvas);
    this.domElement.appendChild(toolsContainer);

    if (audioBuffer) {
      this.load(name || "Untitled", audioBuffer);
    }
  }

  load(name: string, audioBuffer: AudioBuffer) {
    this._canvas.width = audioBuffer.length;
    this._canvas.height = 100;

    this._nameElement.textContent = name;

    GraphicsHelpers.renderWaveform(this._canvas, audioBuffer);
  }
}
