import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { AudioImporter } from "../../elements/src/importers/AudioImporter";
import { getAudioContext } from "../../elements/src/managers/AudioManager";
import { Scene } from "@babylonjs/core/scene";

export interface ISplashScreen {
    image?: string;
    sound?: string;
}

export class SplashScreen {
  audioBuffer?: AudioBuffer;
  mesh: Mesh;

  constructor(
    readonly scene: Scene,
    image: string,
    sound: string
  ) {
    const audioContext = getAudioContext();

    AudioImporter.loadUrl(sound, audioContext).then((buffer) => {
      this.audioBuffer = buffer;
    });

    this.mesh = Mesh.CreatePlane("splash", 1, this.scene);
    this.mesh.billboardMode = Mesh.BILLBOARDMODE_ALL;
    this.mesh.material = new StandardMaterial("mat", this.scene);
    this.mesh.material.backFaceCulling = false;
    this.mesh.material.alpha = 0;

    this.mesh.material.getActiveTextures()[0] = new Texture(
      "data:" + image,
      this.scene
    );
  }
}
