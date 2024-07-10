import { BaseElement } from "../../../elements/src/elements/BaseElement";
import { WindowContainer } from "../../../elements/src/elements/WindowContainer";
import { Project } from "./Project";
import { IKeyboardEvent, Keyboard } from "./components/Keyboard";
import { MixerWindow } from "./windows/MixerWindow";
import { PatternWindow } from "./windows/PatternWindow";
import { PianoRollWindow } from "./windows/PianoRollWindow";
import { ProjectWindow } from "./windows/ProjectWindow";

export class ProjectUI extends BaseElement<"update"> {
  container: WindowContainer;
  readonly pianoRollWindow: PianoRollWindow;
  readonly mixerWindow: MixerWindow;
  patternWindow: PatternWindow;

  get audioContext() {
    return this.project.audioClock.audioContext;
  }

  constructor(readonly project: Project) {
    super("div", "project-ui");

    this.container = new WindowContainer();
    this.domElement.appendChild(this.container.domElement);

    this.pianoRollWindow = new PianoRollWindow(this);
    this.container.addWindow(this.pianoRollWindow);

    this.patternWindow = new PatternWindow(this);
    this.container.addWindow(this.patternWindow);
    this.patternWindow.show(35, 100);

    const playlistWindow = new ProjectWindow(this);
    this.container.addWindow(playlistWindow);
    playlistWindow.show(1000, 50);

    this.mixerWindow = new MixerWindow(this);
    this.container.addWindow(this.mixerWindow);
    this.mixerWindow.show(1000, 470);

    const keyboard = new Keyboard();
    this.domElement.appendChild(keyboard.domElement);
    keyboard.on("update", (event) => {
      const e = event as IKeyboardEvent;
      console.log("Keyboard", event);
      if (e.type === "press") {
        this.patternWindow.trigger(e.note + 24, 0, e.velocity);
      } else if (e.type === "release") {
        this.patternWindow.release(e.note + 24);
      }
    });
  }
}
