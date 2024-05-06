import { GridItem } from "./Grid";

export class Composition {
  title: string;
  description: string;
  tempo: number;
  patterns: GridItem[][] = [[]];

  constructor() {
    this.title = "Untitled";
    this.description = "No description";
    this.tempo = 120;
  }
}
