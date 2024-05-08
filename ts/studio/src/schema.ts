export interface IEventItem {
  note: number;
  start: number;
  velocity: number;
  duration: number;
  label: string;
}

export interface ISequence {
  name: string;
  events: IEventItem[];
}

export interface IPattern {
  name: string;
  tracks: ISequence[];
}

export interface ITimelineItem {
  start: number;
  duration: number;
  label: string;
}

export interface ITimelinePattern extends ITimelineItem {
  pattern: IPattern;
}

export interface ITimelineSequence {
  name: string;
  items: ITimelineItem[];
}

export interface IProject {
  title: string;
  description: string;
  tempo: number;
  patterns: IPattern[];
  timeline: ITimelineSequence[];
}
