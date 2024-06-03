import { BaseElement } from "../../../elements/src/elements/BaseElement";
import { WindowContainer } from "../../../elements/src/elements/WindowContainer";
import { Project } from "./Project";
import { IKeyboardEvent, Keyboard } from "./components/Keyboard";
import { PatternTrack } from "./components/PatternTrack";
import { PatternWindow } from "./windows/PatternWindow";
import { PianoRollWindow } from "./windows/PianoRollWindow";
import { PlaylistWindow } from "./windows/PlaylistWindow";

export class ProjectUI extends BaseElement<"update"> {
  container: WindowContainer;

  constructor(readonly project: Project) {
    super("div", "project-ui");

    this.container = new WindowContainer();
    this.domElement.appendChild(this.container.domElement);

    const pianoRollWindow = new PianoRollWindow(project.audioClock);
    this.container.addWindow(pianoRollWindow);    
    pianoRollWindow.setPosition(500, 100);

    const patternWindow = new PatternWindow(project);    
    patternWindow.show(35, 100);
    this.container.addWindow(patternWindow);

    const playlistWindow = new PlaylistWindow(project);
    patternWindow.show(200, 200);
    this.container.addWindow(playlistWindow);

    patternWindow.on("edit", (selectedTrack) => {
      console.log("edit", selectedTrack);
      pianoRollWindow.loadTrack(selectedTrack as PatternTrack);
      pianoRollWindow.show();
    });

    patternWindow.on("select", (selectedTrack) => {
      console.log("select", selectedTrack);
      const track = selectedTrack as PatternTrack;
      track.instrument.window.show();
    });

    // playlistWindow.show(600, 100);

    // const samplerWindow: SamplerWindow = project.instruments[0] as SamplerWindow;
    // container.addWindow(samplerWindow);
    //samplerWindow.setSize(400, 400);
    //samplerWindow.show(1465, 100);

    const keyboard = new Keyboard();
    this.domElement.appendChild(keyboard.domElement);
    keyboard.on("update", (event) => {
      const e = event as IKeyboardEvent;
      console.log("Keyboard", event);
      if (e.type === "press") {
        patternWindow.trigger(e.note);
      } else if (e.type === "release") {
        console.log("Release", e.note);
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
        for (const instrument of project.instruments) {
          this.container.addWindow(instrument.window);
        }
      }
    });
  }
}
