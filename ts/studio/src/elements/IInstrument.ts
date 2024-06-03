import { IInstrumentSettings } from "../schema";
import { ISourceEvent } from "./components/SamplerSlot";

export interface IInstrument extends IInstrumentSettings {
  trigger(
    note: number,
    when: number,
    velocity: number | undefined
  ): ISourceEvent | void;

  release(note: number): void;
}
