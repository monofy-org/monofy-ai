import { IPattern, IProject, ITimelineSequence } from "../schema";

export class Composition implements IProject {
  title: string;
  description: string;
  tempo: number;
  patterns: IPattern[] = [];
  timeline: ITimelineSequence[] = [];

  constructor() {
    this.title = "Untitled";
    this.description = "No description";
    this.tempo = 120;
  }

  serialize(): string {
    return JSON.stringify(this);
  }
}
