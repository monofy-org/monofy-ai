import { DraggableWindow } from "../../../../elements/src/elements/DraggableWindow";
import { WaveEditor } from "../components/WaveEditor";

export class WaveEditorWindow extends DraggableWindow {
  waveEditor: WaveEditor;

  constructor(name: string, audioBuffer: AudioBuffer) {
    const waveEditor = new WaveEditor(name, audioBuffer);

    super({
      title: "Wave Editor",
      persistent: false,
      width: 800,
      height: 400,
      content: waveEditor.domElement,      
    });

    this.waveEditor = waveEditor;
  }
}
