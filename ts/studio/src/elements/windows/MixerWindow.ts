import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { ProjectUI } from "../ProjectUI";
import { MixerChannel } from "../components/MixerChannel";

export class MixerWindow extends DraggableWindow {
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

    const audioContext = ui.audioContext;

    const masterChannel = new MixerChannel(
      0,
      "Master",
      audioContext.createGain()
    );
    masterChannel.gainNode.connect(audioContext.destination);
    container.appendChild(masterChannel.domElement);

    this.content.appendChild(container);

    for (let i = 1; i <= 16; i++) {
      const channel = new MixerChannel(i, `${i}`, audioContext.createGain());
      channel.gainNode.connect(masterChannel.gainNode);
      container.appendChild(channel.domElement);
    }
  }
}
