import { AudioClock } from "./elements/components/AudioClock";
import { Project } from "./elements/Project";
import { ProjectUI } from "./elements/ProjectUI";
import { FMBass } from "./plugins/fmbass/FMBass";
import { FMPiano } from "./plugins/fmpiano/FMPiano";
import { Plugins } from "./plugins/plugins";
import { Multisampler } from "./plugins/sampler/Multisampler";
import { templates } from "./schema";

Plugins.register("multisampler", Multisampler);
Plugins.register("fm_bass", FMBass);
Plugins.register("fm_piano", FMPiano);

const domElement = document.createElement("div");
domElement.classList.add("studio");

const audioClock = new AudioClock();
const project = new Project(audioClock, templates.Basic);
const projectUI = new ProjectUI(project);

domElement.appendChild(audioClock.domElement);
domElement.appendChild(projectUI.domElement);
document.body.appendChild(domElement);
