import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { ContextMenu } from "../../../../elements/src/elements/ContextMenu";
import { GraphicsHelpers } from "../../abstracts/GraphicsHelpers";
import { TimePositionInput } from "./TimePositionInput";

function pixelsToSeconds(
  canvas: HTMLCanvasElement,
  buffer: AudioBuffer,
  x: number
) {
  return (x / canvas.width) * buffer.duration;
}

function secondsToPixels(
  canvas: HTMLCanvasElement,
  buffer: AudioBuffer,
  seconds: number
) {
  return (seconds / buffer.duration) * canvas.width;
}

export class WaveEditor extends BaseElement<"change" | "add"> {
  _canvas: HTMLCanvasElement;
  private _selection: HTMLDivElement;
  private _dragStart?: number;
  private _startTime: TimePositionInput;
  private _endTime: TimePositionInput;
  private _audioBuffer?: AudioBuffer;
  private _previewSource?: AudioBufferSourceNode;

  get audioBuffer() {
    return this._audioBuffer;
  }

  constructor(
    readonly audioContext: AudioContext,
    name?: string,
    audioBuffer?: AudioBuffer
  ) {
    super("div", "wave-editor");

    const buttonsContainer = document.createElement("div");
    buttonsContainer.classList.add("wave-editor-buttons");
    this.domElement.appendChild(buttonsContainer);

    const toolButtons = document.createElement("div");
    buttonsContainer.appendChild(toolButtons);

    const cropButton = document.createElement("button");
    cropButton.textContent = "Crop";
    toolButtons.appendChild(cropButton);

    const deleteButton = document.createElement("button");
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", () => this._deleteSelection());
    toolButtons.appendChild(deleteButton);

    const cropMenu = new ContextMenu(cropButton);
    cropMenu.addItem("Crop", () => this._crop());
    cropMenu.addItem("Crop As New", () => this._cropAsNew());

    cropButton.addEventListener("click", (e) =>
      cropMenu.show(e.clientX, e.clientY)
    );

    const playStop = document.createElement("div");
    buttonsContainer.appendChild(playStop);

    const playButton = document.createElement("button");
    playButton.textContent = "Play";
    playButton.addEventListener("click", () => this._play());
    playStop.appendChild(playButton);

    const stopButton = document.createElement("button");
    stopButton.textContent = "Stop";
    stopButton.addEventListener("click", () => this._stop());
    playStop.appendChild(stopButton);

    const unused = document.createElement("div");
    buttonsContainer.appendChild(unused);

    const canvasContainer = document.createElement("div");
    canvasContainer.classList.add("wave-editor-canvas-container");
    this.domElement.appendChild(canvasContainer);

    this._canvas = document.createElement("canvas");
    this._canvas.width = 2048;
    this._canvas.height = 200;

    canvasContainer.appendChild(this._canvas);

    const timeContainer = document.createElement("div");
    timeContainer.classList.add("wave-editor-times");
    this.domElement.appendChild(timeContainer);

    const startTimeLabel = document.createElement("label");
    startTimeLabel.textContent = "START:";
    timeContainer.appendChild(startTimeLabel);

    this._startTime = new TimePositionInput();
    timeContainer.appendChild(this._startTime.domElement);
    this._startTime.on("change", () => {
      this._updateSelection();
      if (this._previewSource) {
        this._previewSource.loopStart = this._startTime.value;
      }
    });

    const endTimeLabel = document.createElement("label");
    endTimeLabel.textContent = "END:";
    timeContainer.appendChild(endTimeLabel);

    this._endTime = new TimePositionInput();
    timeContainer.appendChild(this._endTime.domElement);
    this._endTime.on("change", () => {
      this._updateSelection();
      if (this._previewSource) {
        this._previewSource.loopEnd = this._endTime.value;
      }
    });

    this._selection = document.createElement("div");
    this._selection.classList.add("wave-editor-selection");
    canvasContainer.appendChild(this._selection);

    const onmove = (e: PointerEvent) => {
      if (this._dragStart !== undefined && this._audioBuffer) {
        const delta = e.clientX - this._dragStart;
        this._selection.style.width = `${delta}px`;

        this._endTime.value = pixelsToSeconds(
          this._canvas,
          this._audioBuffer,
          e.layerX
        );
      }
    };

    const onup = () => {
      if (this._dragStart !== undefined) {
        this._dragStart = undefined;
        window.removeEventListener("pointermove", onmove);
        window.removeEventListener("pointerup", onup);
        if (this._previewSource) {
          this._previewSource.loopStart = this._startTime.value;
          this._previewSource.loopEnd = this._endTime.value;
        }
      }
    };

    this._canvas.addEventListener("pointerdown", (e) => {
      console.log("DEBUG", e.layerX);

      if (!this._audioBuffer) return;

      if (e.button === 0) {
        this._dragStart = e.clientX;
        this._selection.style.left = `${e.layerX}px`;
        this._selection.style.width = "0px";

        this._startTime.value = pixelsToSeconds(
          this._canvas,
          this._audioBuffer,
          e.layerX
        );

        this._endTime.value = this._startTime.value;

        console.log("pointerdown", e.button);

        window.addEventListener("pointermove", onmove);
        window.addEventListener("pointerup", onup);
      }
    });

    this.on("change", () => {});

    if (audioBuffer) {
      setTimeout(() => {
        this.load(name || "Untitled", audioBuffer);
      }, 1);
    }
  }

  load(name: string, audioBuffer: AudioBuffer) {
    this._audioBuffer = audioBuffer;
    this._draw();
  }

  private _updateSelection() {
    if (!this._audioBuffer) return;

    const start = this._startTime.value;
    const end = this._endTime.value;

    this._selection.style.left = `${secondsToPixels(
      this._canvas,
      this._audioBuffer,
      start
    )}px`;

    this._selection.style.width = `${
      secondsToPixels(this._canvas, this._audioBuffer, end) -
      parseFloat(this._selection.style.left)
    }px`;
  }

  private _getSelectedBuffer(): AudioBuffer | undefined {
    if (!this._audioBuffer) return;

    const start = this._startTime.value;
    const end = this._endTime.value;

    const startFrame = Math.floor(start * this._audioBuffer.sampleRate);
    const endFrame = Math.floor(end * this._audioBuffer.sampleRate);

    const newBuffer = new AudioContext().createBuffer(
      this._audioBuffer.numberOfChannels,
      endFrame - startFrame,
      this._audioBuffer.sampleRate
    );

    for (let i = 0; i < this._audioBuffer.numberOfChannels; i++) {
      const channel = this._audioBuffer.getChannelData(i);
      const newChannel = newBuffer.getChannelData(i);
      let newFrame = 0;
      for (let j = startFrame; j < endFrame; j++) {
        newChannel[newFrame] = channel[j];
        newFrame++;
      }
    }

    return newBuffer;
  }

  private _draw() {
    if (!this._audioBuffer) {
      throw new Error("No audio buffer loaded");
    }
    GraphicsHelpers.renderWaveform(
      this._canvas,
      this._audioBuffer,
      this._canvas.parentElement?.style.color || "#bbaaff"
    );
  }

  private _crop() {
    const newBuffer = this._getSelectedBuffer();
    if (!newBuffer) return;

    this._audioBuffer = newBuffer;
    this._draw();

    this._startTime.value = 0;
    this._endTime.value = 0;

    this._selection.style.width = "0px";

    this.emit("change");
  }

  private _cropAsNew() {
    const newBuffer = this._getSelectedBuffer();
    if (!newBuffer) return;

    this.emit("add", newBuffer);
  }

  private _play() {
    if (!this._audioBuffer) throw new Error("No audio buffer loaded");

    if (this._previewSource) {
      this._previewSource.stop();
      this._previewSource.disconnect();
    }

    const source = this.audioContext.createBufferSource();

    const startTime = this._startTime.value;
    const endTime = this._endTime.value;
    if (endTime < startTime) {
      console.error(endTime, ">", startTime);
      throw new Error("End time is before start time");
    } else if (endTime > startTime) {
      source.loop = true;
      source.loopStart = startTime;
      source.loopEnd = endTime;
    } else {
      source.loop = false;
    }

    this._previewSource = source;
    source.buffer = this._audioBuffer;
    source.connect(this.audioContext.destination);
    source.start(this.audioContext.currentTime, startTime);
  }

  private _stop() {
    if (this._previewSource) {
      this._previewSource.stop();
      this._previewSource.disconnect();
    }
  }

  _deleteSelection() {
    if (!this._audioBuffer) throw new Error("No audio buffer loaded");

    const start = this._startTime.value;
    const end = this._endTime.value;

    const startFrame = Math.floor(start * this._audioBuffer.sampleRate);
    const endFrame = Math.floor(end * this._audioBuffer.sampleRate);

    const newLength = this._audioBuffer.length - (endFrame - startFrame);

    const newBuffer = new AudioContext().createBuffer(
      this._audioBuffer.numberOfChannels,
      newLength,
      this._audioBuffer.sampleRate
    );

    for (let i = 0; i < this._audioBuffer.numberOfChannels; i++) {
      const channel = this._audioBuffer.getChannelData(i);
      const newChannel = newBuffer.getChannelData(i);
      let newFrame = 0;
      for (let j = 0; j < this._audioBuffer.length; j++) {
        if (j < startFrame || j >= endFrame) {
          newChannel[newFrame] = channel[j];
          newFrame++;
        }
      }
    }

    this._audioBuffer = newBuffer;
    this._draw();

    this._endTime.value = this._startTime.value;
    this._updateSelection();

    this.emit("change");
  }
}
