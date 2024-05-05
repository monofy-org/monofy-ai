import { PianoRoll } from "./elements/PianoRoll";
import { AudioClock } from "./elements/AudioClock";

const audioClock = new AudioClock();
const pianoRoll = new PianoRoll(audioClock);

const domElement = document.createElement("div");
domElement.style.width = "100%";
domElement.style.height = "100%";
document.body.appendChild(domElement);

domElement.appendChild(audioClock.domElement);
domElement.appendChild(pianoRoll.domElement);
