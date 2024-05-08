import { AudioClock } from "./elements/components/AudioClock";
import { WindowContainer } from "../../elements/src/elements/WindowContainer";
import { PatternWindow } from "./elements/windows/PatternWindow";
import { PlaylistWindow } from "./elements/windows/PlaylistWindow";
import { Composition } from "./elements/Composition";
import { PatternTrack } from "./elements/components/PatternTrack";
import { PianoRollWindow } from "./elements/windows/PianoRollWindow";

const domElement = document.createElement("div");
domElement.style.display = "flex";
domElement.style.flexDirection = "column";
domElement.style.width = "100%";
domElement.style.height = "100%";

const audioClock = new AudioClock();
domElement.appendChild(audioClock.domElement);

const container = new WindowContainer();
domElement.appendChild(container.domElement);

const composition = new Composition();

const pianoRollWindow = new PianoRollWindow(audioClock);

container.addWindow(pianoRollWindow);
pianoRollWindow.setSize(800, 400);

pianoRollWindow.setPosition(window.innerWidth / 2, 100);

const patternWindow = new PatternWindow(audioClock, composition);
pianoRollWindow.loadTrack(patternWindow.addTrack("Kick"));
patternWindow.addTrack("Snare");
patternWindow.addTrack("Hats");
patternWindow.addTrack("Bass");
patternWindow.addTrack("Lead");
patternWindow.show(100, 100);
container.addWindow(patternWindow);

patternWindow.on("select", (selectedTrack) => {
  console.log("select", selectedTrack);
  pianoRollWindow.loadTrack(selectedTrack as PatternTrack);
  pianoRollWindow.show();
});

const playlistWindow = new PlaylistWindow(audioClock, composition);
for (let i = 1; i <= 9; i++) {
  playlistWindow.addTrack(`Pattern ${i}`);
}
container.addWindow(playlistWindow);
// playlistWindow.show(600, 100);

document.body.appendChild(domElement);

console.log("Serialized", composition.serialize());
