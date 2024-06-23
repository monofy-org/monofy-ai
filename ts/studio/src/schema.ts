export type ControllerGroup = { [key: string]: IPluginControl };
export type ControllerValue = number | string | boolean | AudioBuffer | object;
export type ControlType =
  | "knob"
  | "slider"
  | "string"
  | "select"
  | "audio"
  | "object";

export interface IEvent {
  label: string;
  start: number;
  duration: number;
  note?: number;
  row?: number;
  velocity?: number;
  domElement?: HTMLElement;
}

export interface IPlaylistEvent extends IEvent {
  type: "pattern" | "audio";
  value: IPattern | AudioBuffer;
  domElement?: HTMLElement;
  start: number;
  duration: number;
}

export interface ISequence {
  events: IEvent[];
}

export interface IPattern {
  name: string;
  tracks: ISequence[];
  image?: string;
}

export interface ITrackOptions {
  name: string;
  mute: boolean;
  solo: boolean;
  selected: boolean;
}

export interface IHasOwnEnvelope {
  envelope: IEnvelope | null;
}

export interface IHasOwnLFO {
  lfo?: IOscillatorSettings;
}

export interface IUseEnvelope {
  envelopeNumber: number | null;
}

export interface IUseLFO {
  lfoNumber: number | null;
}

export interface IHasControls {
  controllerGroups: ControllerGroup[];
}

export interface IInstrumentSettings extends IHasControls {
  name: string;
  id: string;
  inputPort?: number;
  inputChannel?: number;
}

export interface IPlaylist {
  events: IPlaylistEvent[];
}

export interface IMixer {
  channels: IMixerChannel[];
}

export interface IEffect extends IHasControls {
  plugin: string;
}

export interface IMixerChannel {
  label: string;
  gain: number;
  mute: boolean;
  solo: boolean;
  outputs: number[];
  effects: IEffect[];
}

export interface IProject {
  title: string;
  description: string;
  tempo: number;
  instruments: IInstrumentSettings[];
  patterns: IPattern[];
  tracks: ITrackOptions[];
  playlist: IPlaylist;
  mixer: IMixer;
}

export interface ISamplerSlot {
  name: string;
  buffer: AudioBuffer | null;
  keyBinding: number | null;
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
}

export interface ISamplerSettings {
  name: string;
  slots: ISamplerSlot[];
}

export interface IPluginControl {
  type: ControlType;
  name: string;
  label: string;
  default: ControllerValue;
  value: ControllerValue;
  min?: number;
  max?: number;
  step?: number;
}

export interface IEnvelope {
  attack: IPluginControl;
  hold: IPluginControl;
  decay: IPluginControl;
  sustain: IPluginControl;
  release: IPluginControl;
}

export interface IOscillatorSettings extends IHasOwnEnvelope {
  shape: "sine" | "square" | "sawtooth" | "triangle";
  detune: number;
  gain?: number;
  route?: number;
  feedback?: number;
}

export interface IFilter extends IHasOwnEnvelope {
  type:
    | "lowpass"
    | "highpass"
    | "bandpass"
    | "lowshelf"
    | "highshelf"
    | "peaking"
    | "notch"
    | "allpass";
  frequency: number;
  detune: number;
  Q: number;
  gain: number;
}

export interface ISynthesizerSettings {
  name: string;
  voices: ISynthesizerVoiceSettings[];
}

export interface ISynthesizerVoiceSettings extends IUseEnvelope, IUseLFO {
  gain: number;
  frequency: number | null;
  detune: number | null;
  oscillators: IOscillatorSettings[];
  note: number | null;
}

export const templates: { [key: string]: IProject } = {
  Basic: {
    title: "Basic",
    description: "Basic project template with Sampler and Synthesizer",
    tempo: 120,
    instruments: [
      { name: "Sampler", id: "sampler", controllerGroups: [] },
      {
        name: "FM Bass",
        id: "fm_bass",
        controllerGroups: [],
      },
    ],
    patterns: [
      {
        name: "Pattern 1",
        tracks: [
          {
            events: [],
          },
          {
            events: [],
          },
        ],
      },
    ],
    tracks: [],
    playlist: { events: [] },
    mixer: {
      channels: [
        {
          label: "Master",
          gain: 1,
          mute: false,
          solo: false,
          outputs: [],
          effects: [],
        },
        {
          label: "Drums",
          gain: 1,
          mute: false,
          solo: false,
          outputs: [0],
          effects: [],
        },
        {
          label: "Bass",
          gain: 1,
          mute: false,
          solo: false,
          outputs: [0],
          effects: [],
        },
      ],
    },
  },
};
