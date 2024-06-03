import { AudioClock } from "./elements/components/AudioClock";
import { WindowContainer } from "../../elements/src/elements/WindowContainer";
import { PatternWindow } from "./elements/windows/PatternWindow";
import { PlaylistWindow } from "./elements/windows/PlaylistWindow";
import { Project } from "./elements/Project";
import { PatternTrack } from "./elements/components/PatternTrack";
import { PianoRollWindow } from "./elements/windows/PianoRollWindow";
import { IKeyboardEvent, Keyboard } from "./elements/components/Keyboard";
import { templates } from "./schema";

const domElement = document.createElement("div");
domElement.style.display = "flex";
domElement.style.flexDirection = "column";
domElement.style.width = "100%";
domElement.style.height = "100%";

const audioClock = new AudioClock();
domElement.appendChild(audioClock.domElement);

const container = new WindowContainer();
domElement.appendChild(container.domElement);

const project = new Project(audioClock);
project.on("update", (event: unknown) => {
  console.log("Composition", event);

  if (!(event instanceof Object)) {
    throw new Error("Invalid event");
  }

  if (!("type" in event)) {
    throw new Error("Invalid event");
  }

  const e = event as { type: string; value: unknown };

  if (e.type === "project") {
    const project = e.value as Project;
    console.log("Project", project);
    patternWindow.loadProject(project);
  }
});

const pianoRollWindow = new PianoRollWindow(audioClock);
container.addWindow(pianoRollWindow);
pianoRollWindow.setSize(900, 400);
pianoRollWindow.setPosition(500, 100);

const patternWindow = new PatternWindow(project);
patternWindow.setSize(400, 400);
patternWindow.show(35, 100);
container.addWindow(patternWindow);

const playlistWindow = new PlaylistWindow(project);
patternWindow.show(200, 200);
container.addWindow(playlistWindow);

patternWindow.on("edit", (selectedTrack) => {
  console.log("edit", selectedTrack);
  pianoRollWindow.loadTrack(selectedTrack as PatternTrack);
  pianoRollWindow.show();
});

// playlistWindow.show(600, 100);

// const samplerWindow: SamplerWindow = project.instruments[0] as SamplerWindow;
// container.addWindow(samplerWindow);
//samplerWindow.setSize(400, 400);
//samplerWindow.show(1465, 100);

const keyboard = new Keyboard();
domElement.appendChild(keyboard.domElement);
keyboard.on("update", (event) => {
  const e = event as IKeyboardEvent;
  console.log("Keyboard", event);
  if (e.type === "press") {
    patternWindow.trigger(e.note);
  } else if (e.type === "release") {
    console.log("Release", e.note);
  }
});

document.body.appendChild(domElement);

project.deserialize(JSON.stringify(templates.Basic));
