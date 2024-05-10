import { triggerActive } from "../../../../elements/src/animation";
import { AudioCanvas } from "../../../../elements/src/elements/AudioCanvas";
import { BaseElement } from "../../../../elements/src/elements/BaseElement";
import { getAudioContext } from "../../../../elements/src/managers/AudioManager";
import { ISamplerSlot } from "../../schema";
import { IInstrument } from "../IInstrument";
import { AudioClock } from "./AudioClock";

interface ISourceEvent {
  source: AudioBufferSourceNode;
  gain: GainNode;
  cutByGroups: number[];
}

export class SamplerSlot
  extends BaseElement<"update">
  implements ISamplerSlot, IInstrument
{
  private _nameElement;

  name: string;
  buffer: AudioBuffer | null = null;
  keyBinding: number | null = null;
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
  cutGroup: number;
  cutByGroups: number[];
  protected _sources: ISourceEvent[] = [];
  private _previewCanvas: AudioCanvas;

  constructor(
    readonly audioClock: AudioClock,
    name = "",
    keyBinding: number | null = null,
    cutGroup = -1,
    cutByGroups: number[] = []
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
    const audioContext = getAudioContext();

    this._nameElement.textContent = "(Loading)";

    const response = await fetch(url);
    if (!response.ok) {
      this._nameElement.textContent = "(Empty)";
      console.error(`Error loading sample: ${url}`, response.statusText);
      return;
    }

    response
      .arrayBuffer()
      .then((buffer) => audioContext.decodeAudioData(buffer))
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

    const audioContext = getAudioContext();
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

    const entry: ISourceEvent = { source, gain, cutByGroups: this.cutByGroups };

    this._sources.push(entry);

    source.onended = () => {
      source.disconnect();
      gain.disconnect();
      const index = this._sources.indexOf(entry);
      if (index >= 0) this._sources.splice(index, 1);
    };

    const time = audioContext.currentTime + (beat * 60) / this.audioClock.bpm;

    if (this.startPosition >= 0) {
      source.start(
        time,
        this.startPosition,
        (this.endPosition || this.buffer.duration) - this.startPosition
      );
    } else {
      source.start(time || audioContext.currentTime);
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
  }

  release(time: number) {
    for (const entry of this._sources) {
      this.cut(entry, time);
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
