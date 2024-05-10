export interface IInstrument {
  name: string;
  trigger(note: number, velocity: number): void;
  release(note: number): void;
}
