export interface IInstrument {
  name: string;
  trigger(note: number, when: number, velocity: number | undefined): void;
  release(note: number): void;
}
