import { AudioClock } from "./elements/components/AudioClock";
import { WindowContainer } from "../../elements/src/elements/WindowContainer";
import { PatternWindow } from "./elements/windows/PatternWindow";
import { PlaylistWindow } from "./elements/windows/PlaylistWindow";
import { Composition } from "./elements/Composition";
import { PatternTrack } from "./elements/components/PatternTrack";
import { PianoRollWindow } from "./elements/windows/PianoRollWindow";
import { SamplerWindow } from "./elements/windows/SamplerWindow";
import { IKeyboardEvent, Keyboard } from "./elements/components/Keyboard";

const composition = new Composition();

const domElement = document.createElement("div");
domElement.style.display = "flex";
domElement.style.flexDirection = "column";
domElement.style.width = "100%";
domElement.style.height = "100%";

const audioClock = new AudioClock();
domElement.appendChild(audioClock.domElement);

const container = new WindowContainer();
domElement.appendChild(container.domElement);

const pianoRollWindow = new PianoRollWindow(audioClock);
container.addWindow(pianoRollWindow);
pianoRollWindow.setSize(900, 400);
pianoRollWindow.setPosition(500, 100);

const patternWindow = new PatternWindow(audioClock, composition);
pianoRollWindow.loadTrack(patternWindow.addTrack("Kick"));
patternWindow.addTrack("Snare");
patternWindow.addTrack("Hats");
patternWindow.addTrack("Bass");
patternWindow.addTrack("Lead");
patternWindow.setSize(400, 400);
patternWindow.show(35, 100);
container.addWindow(patternWindow);

patternWindow.on("select", (selectedTrack) => {
  console.log("select", selectedTrack);
  pianoRollWindow.loadTrack(selectedTrack as PatternTrack);
  pianoRollWindow.show();
});

const playlistWindow = new PlaylistWindow(audioClock, composition);
container.addWindow(playlistWindow);
for (let i = 1; i <= 9; i++) {
  playlistWindow.addTrack(`Pattern ${i}`);
}
// playlistWindow.show(600, 100);

const samplerWindow = new SamplerWindow(audioClock);
container.addWindow(samplerWindow);
samplerWindow.setSize(400, 400);
samplerWindow.show(1465, 100);

const keyboard = new Keyboard();
domElement.appendChild(keyboard.domElement);
keyboard.on("update", (event) => {
  const e = event as IKeyboardEvent;
  console.log("Keyboard", event);
  if (e.type === "press") {
    console.log("Press", e.note);
  } else if (e.type === "release") {
    console.log("Release", e.note);
  }
});

document.body.appendChild(domElement);

console.log("Serialized", composition.serialize());
