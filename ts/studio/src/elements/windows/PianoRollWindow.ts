import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { ProjectUI } from "../ProjectUI";
import { PatternTrack } from "../components/PatternTrack";
import { PianoRoll } from "../components/PianoRoll";

export class PianoRollWindow extends DraggableWindow {
  pianoRoll: PianoRoll;

  constructor(readonly ui: ProjectUI) {
    const pianoRoll = new PianoRoll(ui.project.audioClock);

    super({
      title: "Piano Roll",
      persistent: true,
      content: pianoRoll.domElement,
      width: 900,
      height: 400,
      left: 100,
      top: 100,
    });
    this.pianoRoll = pianoRoll;
  }

  loadTrack(track: PatternTrack) {
    this.pianoRoll.load(track);
    this.setTitle(`Piano Roll (${track.name})`);
  }
}
