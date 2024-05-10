import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { AudioClock } from "../components/AudioClock";
import { PatternTrack } from "../components/PatternTrack";
import { PianoRoll } from "../components/PianoRoll";

export class PianoRollWindow extends DraggableWindow<"update"> {
  pianoRoll: PianoRoll;

  constructor(readonly clock: AudioClock) {
    const pianoRoll = new PianoRoll(clock);

    super("Piano Roll", true, pianoRoll.domElement);
    this.pianoRoll = pianoRoll;

    this.setSize(800, 400);
  }

  loadTrack(track: PatternTrack) {
    this.pianoRoll.load(track);
    this.setTitle(`Piano Roll (${track.name})`);
  }
}
