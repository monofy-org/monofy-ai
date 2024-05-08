import { AudioCursor } from "./components/AudioCursor";

export interface ICursorTimeline {
  domElement: HTMLElement;
  timeline: HTMLElement;
  cursor: AudioCursor;
}
