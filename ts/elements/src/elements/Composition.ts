import { GridItem } from "../../../studio/src/elements/Grid";

export class Composition {
  title: string;
  description: string;
  tempo: number;
  events: GridItem[] = [];

  constructor() {
    this.title = "Untitled";
    this.description = "No description";
    this.tempo = 120;
  }
}
