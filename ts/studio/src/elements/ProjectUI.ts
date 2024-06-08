import { BaseElement } from "../../../elements/src/elements/BaseElement";
import { WindowContainer } from "../../../elements/src/elements/WindowContainer";
import { Project } from "./Project";
import { IKeyboardEvent, Keyboard } from "./components/Keyboard";
import { MixerWindow } from "./windows/MixerWindow";
import { PatternWindow } from "./windows/PatternWindow";
import { PianoRollWindow } from "./windows/PianoRollWindow";
import { PlaylistWindow } from "./windows/PlaylistWindow";

export class ProjectUI extends BaseElement<"update"> {
  container: WindowContainer;
  readonly pianoRollWindow: PianoRollWindow;

  constructor(readonly project: Project) {
    super("div", "project-ui");

    this.container = new WindowContainer();
    this.domElement.appendChild(this.container.domElement);

    this.pianoRollWindow = new PianoRollWindow(this);
    this.container.addWindow(this.pianoRollWindow);

    const patternWindow = new PatternWindow(this);
    this.container.addWindow(patternWindow);
    patternWindow.show(35, 100);

    const playlistWindow = new PlaylistWindow(this);
    this.container.addWindow(playlistWindow);
    playlistWindow.show(1000, 50);

    const mixerWindow = new MixerWindow(this);
    this.container.addWindow(mixerWindow);
    mixerWindow.show(1000, 470);

    const keyboard = new Keyboard();
    this.domElement.appendChild(keyboard.domElement);
    keyboard.on("update", (event) => {
      const e = event as IKeyboardEvent;
      console.log("Keyboard", event);
      if (e.type === "press") {
        patternWindow.trigger(e.note);
      } else if (e.type === "release") {
        patternWindow.release(e.note);
      }
    });

    this.project.on("update", (event) => {
      console.log("ProjectUI project update", event);

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
        playlistWindow.loadProject(project);
        for (const instrument of project.instruments) {
          this.container.addWindow(instrument.window);
        }
      }
    });
  }
  loadProject(project: Project) {
    this.project.load(project);
  }
}
