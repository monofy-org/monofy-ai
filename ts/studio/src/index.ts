import { PianoRoll } from "./elements/PianoRoll";
import { AudioClock } from "./elements/AudioClock";
import { DraggableWindow } from "../../elements/src/elements/DraggableWindow";
import { WindowContainer } from "../../elements/src/elements/WindowContainer";
import { PatternWindow } from "./elements/PatternWindow";
import { PlaylistWindow } from "./elements/PlaylistWindow";
import { Composition } from "./elements/Composition";
import { PatternTrack } from "./elements/PatternTrack";

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

const pianoRoll = new PianoRoll(audioClock);
const pianoRollWindow = new DraggableWindow(
  "Piano Roll",
  true,
  pianoRoll.domElement
);
container.addWindow(pianoRollWindow);
pianoRollWindow.setSize(800, 400);

const patternWindow = new PatternWindow(composition);
patternWindow.addTrack("Kick");
patternWindow.addTrack("Snare");
patternWindow.addTrack("Hats");
patternWindow.addTrack("Bass");
patternWindow.addTrack("Lead");
patternWindow.show(100, 100);
container.addWindow(patternWindow);

patternWindow.on("select", (selectedTrack) => {
  const track = selectedTrack as PatternTrack;
  pianoRoll.loadEvents(track.pattern);
  pianoRollWindow.show();
});

const playlistWindow = new PlaylistWindow(composition);
for (let i = 1; i <= 9; i++) {
  playlistWindow.addTrack(`Pattern ${i}`);
}
container.addWindow(playlistWindow);
playlistWindow.show(600, 100);
pianoRollWindow.show(300, 300);

document.body.appendChild(domElement);
