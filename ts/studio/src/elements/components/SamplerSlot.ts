import { triggerActive } from "../../../../elements/src/animation";
import { AudioCanvas } from "../../../../elements/src/elements/AudioCanvas";
import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { ControllerGroup, ISamplerSlot } from "../../schema";
import { AudioClock } from "./AudioClock";

export interface ISourceEvent {
  source: AudioBufferSourceNode;
  gain: GainNode;
  cutByGroups: number[];
  time: number;
}

export class SamplerSlot extends BaseElement<"update"> implements ISamplerSlot {
  private _nameElement;
  controllerGroups: ControllerGroup[] = [];
  buffer: AudioBuffer | null = null;
  velocity: number;
  pan: number;
  pitch: number;
  randomPitch: number;
  randomVelocity: number;
  startPosition: number;
  endPosition: number;
  loop: boolean;
  loopStart: number;
  loopEnd: number;
  protected _sources: ISourceEvent[] = [];
  private _previewCanvas: AudioCanvas;

  constructor(
    readonly audioClock: AudioClock,
    public name = "",
    public keyBinding: number | null = null,
    public cutGroup = -1,
    readonly cutByGroups: number[] = []
  ) {
    super("div", "sampler-slot");

    this.name = name;
    this.buffer = null;
    this.keyBinding = keyBinding || null;
    this.velocity = 1;
    this.pan = 1;
    this.pitch = 1;
    this.randomPitch = 0;
    this.randomVelocity = 0.05;
    this.startPosition = 0;
    this.endPosition = 0;
    this.loop = false;
    this.loopStart = 0;
    this.loopEnd = 0;
    this.cutGroup = cutGroup;
    this.cutByGroups = cutByGroups;

    this._previewCanvas = new AudioCanvas(200, 60);
    this.domElement.appendChild(this._previewCanvas.domElement);

    this._nameElement = document.createElement("div");
    this._nameElement.classList.add("sampler-slot-name");
    this.domElement.appendChild(this._nameElement);

    this.domElement.addEventListener("pointerdown", () => this.trigger());
  }

  async loadSample(name: string, url: string) {
    this._nameElement.textContent = "(Loading)";

    const response = await fetch(url);
    if (!response.ok) {
      this._nameElement.textContent = "(Empty)";
      console.error(`Error loading sample: ${url}`, response.statusText);
      return;
    }

    response
      .arrayBuffer()
      .then((buffer) => this.audioClock.audioContext.decodeAudioData(buffer))
      .then((audioBuffer) => {
        this._nameElement.textContent = name;
        this.buffer = audioBuffer;
        this.emit("update");
        setTimeout(() => {
          this._previewCanvas.loadBuffer(audioBuffer);
        }, 1);
      })
      .catch((error) => {
        this._nameElement.textContent = "(Empty)";
        console.error(`Error loading sample: ${url}`, error);
      });
  }

  trigger(beat = 0) {
    if (!this.buffer) {
      console.error("No buffer loaded for sample", this.name);
      return;
    }

    const audioContext = this.audioClock.audioContext;
    const source = audioContext.createBufferSource();
    source.buffer = this.buffer;
    source.playbackRate.value = this.pitch + Math.random() * this.randomPitch;
    source.connect(audioContext.destination);

    const gain = audioContext.createGain();
    gain.gain.value = this.velocity + Math.random() * this.randomVelocity;
    gain.connect(audioContext.destination);

    source.connect(gain);

    if (this.pan !== 1) {
      const pan = audioContext.createStereoPanner();
      pan.pan.value = this.pan;
      source.connect(pan);
      pan.connect(gain);
    }

    const time =
      (this.audioClock.startTime || audioContext.currentTime) +
      (beat * 60) / this.audioClock.bpm;

    const entry: ISourceEvent = {
      source,
      gain,
      cutByGroups: this.cutByGroups,
      time,
    };

    this._sources.push(entry);

    source.onended = () => {
      source.disconnect();
      gain.disconnect();
      const index = this._sources.indexOf(entry);
      if (index >= 0) this._sources.splice(index, 1);
    };

    console.log(beat, audioContext.currentTime, time);

    if (this.startPosition >= 0) {
      source.start(
        time,
        this.startPosition,
        (this.endPosition || this.buffer.duration) - this.startPosition
      );
    } else {
      source.start(time);
    }

    if (time === 0 || !this.audioClock.isPlaying) {
      triggerActive(this.domElement);
    } else {
      this.audioClock.scheduleEventAtTime(
        () => triggerActive(this.domElement),
        time
      );
    }

    if (this.loop) {
      source.loop = true;
      source.loopStart = this.loopStart;
      source.loopEnd = this.loopEnd || this.buffer.duration;
    }

    return entry;
  }

  release(time: number) {
    for (const entry of this._sources) {
      if (time >= entry.time) this.cut(entry, time);
    }
  }

  cut(entry: ISourceEvent, time = 0) {
    const currentValue = entry.gain.gain.value;
    entry.gain.gain.cancelScheduledValues(time);
    entry.gain.gain.setValueAtTime(currentValue, time);
    entry.gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.1);
    entry.source.stop(time + 0.1);
  }
}
