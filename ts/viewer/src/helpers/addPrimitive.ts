import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { Scene } from "@babylonjs/core/scene";

export function addPrimitive(
  scene: Scene,
  type: "box" | "sphere" | "cylinder" | "torus" | "plane",
  position: Vector3,
  rotation: Vector3,
  size: Vector3
) {
  let mesh: Mesh;
  switch (type) {
    case "box":
      mesh = MeshBuilder.CreateBox("box", { size: 1 }, scene);
      break;
    case "sphere":
      mesh = MeshBuilder.CreateSphere("sphere", { diameter: 1 }, scene);
      break;
    case "cylinder":
      mesh = MeshBuilder.CreateCylinder(
        "cylinder",
        { diameterTop: 1, diameterBottom: 1, height: 1 },
        scene
      );
      break;
    case "torus":
      mesh = MeshBuilder.CreateTorus(
        "torus",
        { diameter: 1, thickness: 0.5 },
        scene
      );
      break;
    case "plane":
      mesh = MeshBuilder.CreatePlane("plane", { size: 1 }, scene);
      break;
    default:
      throw new Error("Invalid primitive type");
  }

  mesh.position.copyFrom(position);
  mesh.rotation.copyFrom(rotation);
  mesh.scaling.copyFrom(size);
  // mesh.material = this._defaultMaterial;

  return mesh;
}
