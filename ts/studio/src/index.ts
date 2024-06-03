import { AudioClock } from "./elements/components/AudioClock";
import { Project } from "./elements/Project";
import { ProjectUI } from "./elements/ProjectUI";
import { templates } from "./schema";

const domElement = document.createElement("div");
domElement.classList.add("studio");

const audioClock = new AudioClock();
const project = new Project(audioClock);
const projectUI = new ProjectUI(project);

domElement.appendChild(audioClock.domElement);
domElement.appendChild(projectUI.domElement);
document.body.appendChild(domElement);

project.deserialize(JSON.stringify(templates.Basic));
