import { Instrument } from "../../abstracts/Instrument";
import { InstrumentWindow } from "../../abstracts/InstrumentWindow";
import { Project } from "../../elements/Project";
import { Envelope } from "../../elements/components/Envelope";
import { AudioImporter } from "../../../../elements/src/importers/AudioImporter";
import { ISamplerSettings } from "../../schema";

export class Sampler extends Instrument {
  readonly name = "Sampler";
  readonly id = "sampler";
  readonly author = "Johnny Street";
  readonly version = "0.0.1";
  readonly description = "A simple sampler";
  private readonly _sources: {
    source: AudioBufferSourceNode;
    gain: GainNode;
  }[] = [];
  private readonly _gainEnvelope: Envelope;
  private readonly _enableEnvelope = false;
  private _cutBySelf = true;
  private _volume = 0.9;
  public cutGroup = 0;
  public cutByGroups: number[] = [];
  private _window?: InstrumentWindow;

  readonly Window = InstrumentWindow;

  constructor(
    project: Project,
    readonly settings: ISamplerSettings = {}
  ) {
    super(project, settings.mixerChannel);

    this._gainEnvelope = new Envelope({
      attack: settings.envelope?.attack ?? 0,
      hold: settings.envelope?.hold ?? 0,
      decay: settings.envelope?.decay ?? 0,
      sustain: settings.envelope?.sustain ?? 1,
      release: settings.envelope?.release ?? 0,
    });
  }

  trigger(note: number, when: number, velocity: number) {
    if (!this.settings.buffer) {
      console.warn("No buffer loaded");
      return;
    }

    if (this._cutBySelf) {
      this.stop();
    }

    const gain = this.project.audioClock.audioContext.createGain();
    gain.connect(this.output);
    const source = this.project.audioClock.audioContext.createBufferSource();
    source.buffer = this.settings.buffer;
    source.connect(gain);
    const item = { source, gain };
    this._sources.push(item);
    source.onended = () => {
      item.gain.disconnect();
      this._sources.splice(this._sources.indexOf(item), 1);
    };
    source.start(when);
    if (this._enableEnvelope) {
      this._gainEnvelope.trigger(gain.gain, when, velocity);
    }
  }

  release(note: number, when: number): void {
    if (this._enableEnvelope) {
      for (const item of this._sources) {
        this._gainEnvelope.triggerRelease(item.gain.gain, when);
      }
    }
  }

  stop(when = 0) {
    for (const item of this._sources) {
      item.source.stop(when);
    }
  }

  loadUrl(url: string) {
    AudioImporter.loadUrl(url, this.project.audioClock.audioContext).then(
      (buffer) => {
        this.settings.buffer = buffer;
      }
    );
  }

  loadFile(file: File) {
    AudioImporter.loadFile(file, this.project.audioClock.audioContext).then(
      (buffer) => {
        this.settings.buffer = buffer;
      }
    );
  }

  loadBlob(blob: Blob) {
    AudioImporter.loadBlob(blob, this.project.audioClock.audioContext).then(
      (buffer) => {
        this.settings.buffer = buffer;
      }
    );
  }
}
