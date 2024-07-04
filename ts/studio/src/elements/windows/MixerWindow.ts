import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { SelectableGroup } from "../../../../elements/src/elements/SelectableGroup";
import type { ProjectUI } from "../ProjectUI";
import { MixerChannelStrip } from "../components/MixerChannelStrip";

export class MixerWindow extends DraggableWindow {

  readonly channels: SelectableGroup<MixerChannelStrip> = new SelectableGroup();

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
    container.classList.add("mixer-container");

    const channelsContainer = document.createElement("div");
    channelsContainer.classList.add("mixer-channels-container");

    const effectsContainer = document.createElement("div");
    effectsContainer.classList.add("mixer-effects-container");

    container.appendChild(channelsContainer);
    container.appendChild(effectsContainer);

    this.content.appendChild(container);

    console.log("MIXER CHANNELS DEBUG", this.mixer.channels);

    for (let i = 0; i < this.mixer.channels.length; i++) {
      const channel = new MixerChannelStrip(this.channels, this.mixer.channels[i], i === 0);
      this.channels.addSelectable(channel);
      this.mixer.channels[i].gainNode.gain.value = channel.volume;
      channel.on("change", () => {
        this.mixer.channels[i].gainNode.gain.value = channel.volume;
      });
      channelsContainer.appendChild(channel.domElement);
    }

    this.channels.items[0].selected = true;
  }
}
