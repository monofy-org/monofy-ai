import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import type { ProjectUI } from "../ProjectUI";
import { MixerChannelStrip } from "../components/MixerChannelStrip";

export class MixerWindow extends DraggableWindow {
  get mixer() {
    return this.ui.project.mixer;
  }

  constructor(readonly ui: ProjectUI) {
    super({
      title: "Mixer",
      persistent: true,
      content: document.createElement("div"),
      width: 500,
      height: 300,
    });

    this.domElement.classList.add("mixer-window");

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.overflowX = "auto";

    this.content.appendChild(container);

    console.log("MIXER CHANNELS DEBUG", this.mixer.channels);

    for (let i = 0; i < this.mixer.channels.length; i++) {
      const channel = new MixerChannelStrip(this.mixer.channels[i], i === 0);
      this.mixer.channels[i].gainNode.gain.value = channel.volume;
      channel.on("change", () => {
        this.mixer.channels[i].gainNode.gain.value = channel.volume;
      });
      container.appendChild(channel.domElement);
    }
  }
}
