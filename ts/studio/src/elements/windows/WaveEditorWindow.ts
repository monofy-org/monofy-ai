import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { ProjectUI } from "../ProjectUI";
import { WaveEditor } from "../components/WaveEditor";

export class WaveEditorWindow extends DraggableWindow {
  waveEditor: WaveEditor;

  constructor(ui: ProjectUI, item: { name: string; buffer: AudioBuffer }) {
    const waveEditor = new WaveEditor(ui.audioContext, item.name, item.buffer);

    super(ui.container, {
      title: "Wave Editor - " + item.name,
      persistent: false,
      width: 640,
      height: 300,
      content: waveEditor.domElement,
    });

    this.waveEditor = waveEditor;

    this.waveEditor.on("change", () => {
      if (!waveEditor.audioBuffer) throw new Error("No audio buffer");
      item.buffer = waveEditor.audioBuffer;
    });

    this.content.appendChild(waveEditor.domElement);
  }
}
