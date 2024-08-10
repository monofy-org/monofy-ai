import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { ContextMenu } from "../../../elements/src/elements/ContextMenu";
import type { Viewer } from "../viewer";
import { addPrimitive } from "../helpers/addPrimitive";

export class PrimitivesMenu extends ContextMenu {
  constructor(viewer: Viewer) {
    super(document.body);
    this.addItem("Box", () => {
      addPrimitive(
        viewer.scene,
        "box",
        viewer.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });
    this.addItem("Sphere", () => {
      addPrimitive(
        viewer.scene,
        "sphere",
        viewer.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });
    this.addItem("Cylinder", () => {
      addPrimitive(
        viewer.scene,
        "cylinder",
        viewer.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });
    this.addItem("Torus", () => {
      addPrimitive(
        viewer.scene,
        "torus",
        viewer.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });
    this.addItem("Plane", () => {
      addPrimitive(
        viewer.scene,
        "plane",
        viewer.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });
  }
}
