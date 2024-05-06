import { PianoRoll } from "./elements/PianoRoll";
import { AudioClock } from "./elements/AudioClock";
import { DraggableWindow } from "../../elements/src/elements/DraggableWindow";
import { WindowContainer } from "../../elements/src/elements/WindowContainer";

const domElement = document.createElement("div");
domElement.style.display = "flex";
domElement.style.flexDirection = "column";
domElement.style.width = "100%";
domElement.style.height = "100%";

const audioClock = new AudioClock();
domElement.appendChild(audioClock.domElement);

const container = new WindowContainer();
domElement.appendChild(container.domElement);

const pianoRoll = new PianoRoll(audioClock);
const pianoRollWindow = new DraggableWindow(
  "Piano Roll",
  true,
  pianoRoll.domElement
);
container.addWindow(pianoRollWindow);
pianoRollWindow.setSize(800, 400);
pianoRollWindow.show(100, 100);

document.body.appendChild(domElement);
