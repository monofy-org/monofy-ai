import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { ContextMenu } from "../../../../elements/src/elements/ContextMenu";
import { GraphicsHelpers } from "../../abstracts/GraphicsHelpers";
import { TimePositionInput } from "./TimePositionInput";


function pixelsToSeconds(
  canvas: HTMLCanvasElement,
  buffer: AudioBuffer,
  cssPixels: number
): number {
  const displayWidth = canvas.getBoundingClientRect().width;
  // Avoid division by zero if the canvas isn't rendered
  if (displayWidth === 0 || buffer.duration === 0) return 0;
  // Calculate the ratio between the canvas's internal resolution and its displayed size
  const scale = canvas.width / displayWidth;
  // Convert CSS pixels from the mouse event to buffer pixels
  const bufferPixels = cssPixels * scale;
  return (bufferPixels / canvas.width) * buffer.duration;
}

function secondsToPixels(
  canvas: HTMLCanvasElement,
  buffer: AudioBuffer,
  seconds: number
): number {
  const displayWidth = canvas.getBoundingClientRect().width;
  // Avoid division by zero
  if (displayWidth === 0 || buffer.duration === 0) return 0;
  const scale = canvas.width / displayWidth;
  // Calculate the position in the canvas's internal resolution (buffer pixels)
  const bufferPixels = (seconds / buffer.duration) * canvas.width;
  // Convert buffer pixels back to CSS pixels for styling
  return bufferPixels / scale;
}

export class WaveEditor extends BaseElement<"change" | "add"> {
  _canvas: HTMLCanvasElement;
  private _selection: HTMLDivElement;
  private _dragStartOffsetX?: number;
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
      if (this._dragStartOffsetX !== undefined && this._audioBuffer) {
        const currentOffsetX = e.offsetX;

        const startPixel = Math.min(this._dragStartOffsetX, currentOffsetX);
        const endPixel = Math.max(this._dragStartOffsetX, currentOffsetX);

        this._selection.style.left = `${startPixel}px`;
        this._selection.style.width = `${endPixel - startPixel}px`;

        this._startTime.value = pixelsToSeconds(
          this._canvas,
          this._audioBuffer,
          startPixel
        );
        this._endTime.value = pixelsToSeconds(
          this._canvas,
          this._audioBuffer,
          endPixel
        );
      }
    };

    const onup = () => {
      if (this._dragStartOffsetX !== undefined) {
        this._dragStartOffsetX = undefined;
        window.removeEventListener("pointermove", onmove);
        window.removeEventListener("pointerup", onup);
        if (this._previewSource) {
          this._previewSource.loopStart = this._startTime.value;
          this._previewSource.loopEnd = this._endTime.value;
        }
      }
    };

    this._canvas.addEventListener("pointerdown", (e) => {
      if (!this._audioBuffer) return;

      if (e.button === 0) {
        // Use offsetX, which is relative to the canvas's padding edge.
        const startOffsetX = e.offsetX;

        this._dragStartOffsetX = startOffsetX;
        this._selection.style.left = `${startOffsetX}px`;
        this._selection.style.width = "0px";

        const startTime = pixelsToSeconds(
          this._canvas,
          this._audioBuffer,
          startOffsetX
        );
        this._startTime.value = startTime;
        this._endTime.value = startTime;

        window.addEventListener("pointermove", onmove);
        window.addEventListener("pointerup", onup);
      }
    });

    this.on("change", () => { });

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

    const startPixel = secondsToPixels(this._canvas, this._audioBuffer, start);
    const endPixel = secondsToPixels(this._canvas, this._audioBuffer, end);

    this._selection.style.left = `${startPixel}px`;
    this._selection.style.width = `${endPixel - startPixel}px`;
  }

  private _getSelectedBuffer(): AudioBuffer | undefined {
    if (!this._audioBuffer) return;

    // Ensure start is always before end for cropping/deleting
    const start = Math.min(this._startTime.value, this._endTime.value);
    const end = Math.max(this._startTime.value, this._endTime.value);

    const startFrame = Math.floor(start * this._audioBuffer.sampleRate);
    const endFrame = Math.floor(end * this._audioBuffer.sampleRate);

    // If the selection has no duration, do nothing.
    if (startFrame === endFrame) return;

    const newBuffer = this.audioContext.createBuffer(
      this._audioBuffer.numberOfChannels,
      endFrame - startFrame,
      this._audioBuffer.sampleRate
    );

    for (let i = 0; i < this._audioBuffer.numberOfChannels; i++) {
      const channel = this._audioBuffer.getChannelData(i);
      const newChannel = newBuffer.getChannelData(i);
      newChannel.set(channel.subarray(startFrame, endFrame));
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
    this._selection.style.left = "0px";

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

    const startTime = Math.min(this._startTime.value, this._endTime.value);
    const endTime = Math.max(this._startTime.value, this._endTime.value);

    if (endTime > startTime) {
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

    const start = Math.min(this._startTime.value, this._endTime.value);
    const end = Math.max(this._startTime.value, this._endTime.value);

    const startFrame = Math.floor(start * this._audioBuffer.sampleRate);
    const endFrame = Math.floor(end * this._audioBuffer.sampleRate);

    // If the selection has no duration, do nothing.
    if (startFrame === endFrame) return;

    const newLength = this._audioBuffer.length - (endFrame - startFrame);

    const newBuffer = this.audioContext.createBuffer(
      this._audioBuffer.numberOfChannels,
      newLength,
      this._audioBuffer.sampleRate
    );

    for (let i = 0; i < this._audioBuffer.numberOfChannels; i++) {
      const oldChannel = this._audioBuffer.getChannelData(i);
      const newChannel = newBuffer.getChannelData(i);

      // Copy the part before the selection
      newChannel.set(oldChannel.subarray(0, startFrame), 0);
      // Copy the part after the selection
      newChannel.set(oldChannel.subarray(endFrame), startFrame);
    }

    this._audioBuffer = newBuffer;
    this._draw();

    this._endTime.value = this._startTime.value;
    this._updateSelection();

    this.emit("change");
  }
}