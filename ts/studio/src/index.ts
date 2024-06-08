import { AudioClock } from "./elements/components/AudioClock";
import { Project } from "./elements/Project";
import { ProjectUI } from "./elements/ProjectUI";
import { FMBass } from "./plugins/fmbass/FMBass";
import { Plugins } from "./plugins/plugins";
import { Sampler } from "./plugins/sampler/Sampler";
import { templates } from "./schema";

Plugins.register("sampler", Sampler);
Plugins.register("fm_bass", FMBass);

const domElement = document.createElement("div");
domElement.classList.add("studio");

const audioClock = new AudioClock();
const project = new Project(audioClock, templates.Basic);
const projectUI = new ProjectUI(project);

domElement.appendChild(audioClock.domElement);
domElement.appendChild(projectUI.domElement);
document.body.appendChild(domElement);
