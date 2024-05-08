import { IPattern, IProject, ITimelineSequence } from "../schema";

export class Composition implements IProject {
  title: string;
  description: string;
  tempo: number;
  patterns: IPattern[] = [];
  timeline: ITimelineSequence[] = [];

  constructor(title = "Untitled", description = "") {
    this.title = title;
    this.description = description;
    this.tempo = 120;
  }

  serialize(): string {
    return JSON.stringify(this);
  }
}
