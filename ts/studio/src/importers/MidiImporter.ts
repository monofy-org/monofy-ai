import { parseArrayBuffer } from "midi-json-parser";

export abstract class MidiImporter {
  static async importFile(file: File) {
    const arrayBuffer = await file.arrayBuffer();
    return MidiImporter.import(arrayBuffer);
  }

  static import(arrayBuffer: ArrayBuffer) {
    const midi = parseArrayBuffer(arrayBuffer);
    return midi;
  }
}
