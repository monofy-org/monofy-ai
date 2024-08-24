import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { Scene } from "@babylonjs/core/scene";

export class Character extends Mesh {

  constructor(
    readonly name: string,
    readonly scene: Scene,    
  ) {
    super(name, scene);
    this.billboardMode = Mesh.BILLBOARDMODE_ALL;

    this.material = new StandardMaterial("mat", this.scene);
    this.material.backFaceCulling = false;
    this.material.alpha = 0;
  }
  setTexture(url: string) {
    this.material!.getActiveTextures()[0] = new Texture(url, this.scene);
  }
}
