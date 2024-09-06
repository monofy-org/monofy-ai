import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { Scene } from "@babylonjs/core/scene";

export class TerrainHelper {
  static loadTerrainFromHeightmap(
    scene: Scene,
    heightmapUrl: string,
    scale = 256
  ) {
    if (!scene) {
      throw new Error("Scene not initialized");
    }

    console.log("Loading terrain from heightmap", heightmapUrl);

    const terrain = MeshBuilder.CreateGroundFromHeightMap(
      "terrain",
      heightmapUrl,
      {
        width: scale,
        height: scale,
      },
      scene
    );

    terrain.material = new StandardMaterial("terrain", scene);

    return terrain;
  }

  static createDefaultTerrain(scene: Scene, scale = 256) {
    if (!scene) {
      throw new Error("Scene not initialized");
    }

    console.log("Creating default terrain");

    const terrain = MeshBuilder.CreateGround(
      "terrain",
      {
        width: scale,
        height: scale,
      },
      scene
    );

    return terrain;
  }
}
