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
  readonly playlistWindow: ProjectWindow;
  patternWindow: PatternWindow;

  get audioContext() {
    return this.project.audioClock.audioContext;
  }

  constructor(readonly project: Project) {
    super("div", "project-ui");

    this.container = new WindowContainer();
    this.domElement.appendChild(this.container.domElement);

    this.pianoRollWindow = new PianoRollWindow(this);

    this.patternWindow = new PatternWindow(this);
    this.patternWindow.show(35, 100);

    this.playlistWindow = new ProjectWindow(this);
    this.playlistWindow.show(1000, 50);

    this.mixerWindow = new MixerWindow(this);
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

    // handle file drop
    this.domElement.addEventListener("dragover", (e) => {
      e.preventDefault();
    });

    this.domElement.addEventListener("drop", async (e: DragEvent) => {
      e.preventDefault();
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        for (let i = 0; i < files.length; i++) {
          this.playlistWindow.treeView.loadFile(files[i]);
        }
      }
    });
  }
}
