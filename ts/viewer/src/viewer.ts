import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { Engine } from "@babylonjs/core/Engines/engine";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { Texture } from "@babylonjs/core/Materials/Textures/texture";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { PointLight } from "@babylonjs/core/Lights/pointLight";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { Scene } from "@babylonjs/core/scene";
import { PointerEventTypes } from "@babylonjs/core/Events/pointerEvents";
import { Nullable } from "@babylonjs/core/types";
import { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import { GizmoManager } from "@babylonjs/core/Gizmos/gizmoManager";
import { PositionGizmo } from "@babylonjs/core/Gizmos/positionGizmo";
import { RotationGizmo } from "@babylonjs/core/Gizmos/rotationGizmo";
import "@babylonjs/core/Helpers/sceneHelpers";
import { ScaleGizmo } from "@babylonjs/core/Gizmos/scaleGizmo";
import { BoundingBoxGizmo } from "@babylonjs/core/Gizmos/boundingBoxGizmo";
import { GroundMesh } from "@babylonjs/core/Meshes/groundMesh";
import { ContextMenu } from "./elements/ContextMenu";
//import { Inspector } from "@babylonjs/inspector";

const win = window as any;

export class Viewer {
  private readonly _canvas: HTMLCanvasElement;
  private readonly _engine: Engine;
  private readonly _scene: Scene;
  private readonly _camera: ArcRotateCamera;
  private readonly _cursor: Mesh;
  private readonly _defaultMaterial;
  private readonly _contextMenu: ContextMenu;
  private _selectedMesh: Nullable<AbstractMesh> = null;
  public readonly gizmoManager: GizmoManager;
  public readonly cursorPosition: Vector3 = Vector3.Zero();

  ground: GroundMesh;

  constructor(canvasElement?: HTMLCanvasElement) {
    if (canvasElement === undefined) {
      this._canvas = document.createElement("canvas");
      document.body.appendChild(this._canvas);
    } else {
      this._canvas = canvasElement;
    }

    this._canvas.width = win.innerWidth;
    this._canvas.height = win.innerHeight;
    win.addEventListener("resize", () => {
      this._canvas.width = win.innerWidth;
      this._canvas.height = win.innerHeight;
    });

    this._contextMenu = new ContextMenu(document.body, this._canvas);
    const primitiveMenu = new ContextMenu();

    primitiveMenu.addItem("Box", () => {
      this.addPrimitive(
        "box",
        this.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });
    primitiveMenu.addItem("Sphere", () => {
      this.addPrimitive(
        "sphere",
        this.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });
    primitiveMenu.addItem("Cylinder", () => {
      this.addPrimitive(
        "cylinder",
        this.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });
    primitiveMenu.addItem("Torus", () => {
      this.addPrimitive(
        "torus",
        this.cursorPosition,
        Vector3.Zero(),
        new Vector3(1, 1, 1)
      );
    });

    this._contextMenu.addSubmenu("Add Primitive", primitiveMenu);

    this._engine = new Engine(this._canvas, true);
    this._scene = new Scene(this._engine);
    // Inspector.Show(this._scene, {
    //   enableClose: false,
    //   enablePopup: false,
    //   overlay: true,
    // });
    this._cursor = MeshBuilder.CreateSphere(
      "cursor",
      { diameter: 0.1, segments: 16 },
      this._scene
    );

    const cursorMaterial = new StandardMaterial("cursorMaterial", this._scene);
    cursorMaterial.diffuseColor = new Color3(0, 1, 0); // Green color
    cursorMaterial.specularColor = new Color3(0, 0, 0); // No shininess
    this._cursor.material = cursorMaterial;
    this._cursor.material.alpha = 0.5;
    this._cursor.isVisible = false;

    this._defaultMaterial = new StandardMaterial(
      "defaultMaterial",
      this._scene
    );
    this._defaultMaterial.diffuseColor = new Color3(0.8, 0.8, 0.8); // Off white color
    this._defaultMaterial.specularColor = new Color3(0, 0, 0); // No shininess
    this._camera = new ArcRotateCamera(
      "camera",
      Math.PI / 2,
      Math.PI / 2,
      10,
      new Vector3(0, 1.65, 0),
      this._scene
    );
    this._camera.attachControl(this._canvas, true);

    const light = new HemisphericLight(
      "light",
      new Vector3(0, 1, 0),
      this._scene
    );
    light.intensity = 0.7;

    const pointLight = new PointLight(
      "pointLight",
      new Vector3(0, 10, 0),
      this._scene
    );
    pointLight.intensity = 0.7;

    this.gizmoManager = this._createGizmos();
    this.ground = this._createGround();

    this._engine.runRenderLoop(() => {
      this._scene.render();
    });
  }

  public get domElement() {
    return this._canvas;
  }

  private _createGround(width: number = 10, height: number = 10): GroundMesh {
    if (this.ground) {
      this.ground.dispose();
    }

    const ground = MeshBuilder.CreateGround(
      "ground",
      { width, height },
      this._scene
    );
    ground.material = this._defaultMaterial;

    return ground;
  }

  private _createGizmos(): GizmoManager {
    if (this.gizmoManager) {
      throw new Error("Gizmo manager already created");
    }

    win.addEventListener("keydown", (event: KeyboardEvent) => {
      if (event.key === "Delete") {
        if (this._selectedMesh) {
          gizmoManager.attachToMesh(null);
          this._selectedMesh.dispose();
        }
      } else if (event.key === "e" && this._selectedMesh !== this._cursor) {
        if (this._selectedMesh) {
          gizmoManager.rotationGizmoEnabled = true;
          gizmoManager.positionGizmoEnabled = false;
          gizmoManager.scaleGizmoEnabled = false;
          gizmoManager.boundingBoxGizmoEnabled = false;
        }
      } else if (event.key === "w") {
        if (this._selectedMesh) {
          gizmoManager.positionGizmoEnabled = true;
          gizmoManager.rotationGizmoEnabled = false;
          gizmoManager.scaleGizmoEnabled = false;
          gizmoManager.boundingBoxGizmoEnabled = false;
        }
      } else if (event.key === "q" && this._selectedMesh !== this._cursor) {
        if (this._selectedMesh) {
          gizmoManager.positionGizmoEnabled = true;
          gizmoManager.rotationGizmoEnabled = true;
          gizmoManager.boundingBoxGizmoEnabled = true;
          gizmoManager.scaleGizmoEnabled = false;
        }
      } else if (event.key === "s" && this._selectedMesh !== this._cursor) {
        if (this._selectedMesh) {
          gizmoManager.positionGizmoEnabled = false;
          gizmoManager.rotationGizmoEnabled = false;
          gizmoManager.scaleGizmoEnabled = true;
          gizmoManager.boundingBoxGizmoEnabled = false;
        }
      } else if (event.key === "r" && this._selectedMesh !== this._cursor) {
        if (this._selectedMesh) {
          gizmoManager.boundingBoxGizmoEnabled = true;
          gizmoManager.positionGizmoEnabled = false;
          gizmoManager.rotationGizmoEnabled = false;
          gizmoManager.scaleGizmoEnabled = false;
        }
      } else if (event.key === "c") {
        this._placeCursor();
      } else if (event.key == "Escape") {
        if (this._contextMenu.isVisible()) {
          this._contextMenu.hide();
        } else {
          this.deselectAll();
        }
      }
    });

    this._scene.onPointerObservable.add((eventData) => {
      if (eventData.event.button === 2) {
        this._placeCursor();
        return;
      }

      if (eventData.event.button !== 0) {
        return;
      }

      if (eventData.type === PointerEventTypes.POINTERDOUBLETAP) {
        if (eventData.pickInfo?.hit && eventData.pickInfo.pickedMesh) {
          this.selectObject(eventData.pickInfo.pickedMesh);
        } else {
          this.deselectAll();
        }
      } else if (eventData.type === PointerEventTypes.POINTERDOWN) {
        if (eventData.pickInfo?.hit && eventData.pickInfo.pickedMesh) {
          if (eventData.pickInfo.pickedMesh !== this._selectedMesh) {
            this.deselectAll();
          }
        } else {
          this.deselectAll();
        }
      }
    });

    const gizmoManager = new GizmoManager(this._scene);
    gizmoManager.usePointerToAttachGizmos = false;
    gizmoManager.gizmos.positionGizmo = new PositionGizmo();
    gizmoManager.gizmos.rotationGizmo = new RotationGizmo();
    gizmoManager.gizmos.scaleGizmo = new ScaleGizmo();
    gizmoManager.gizmos.boundingBoxGizmo = new BoundingBoxGizmo();
    gizmoManager.positionGizmoEnabled = false;
    gizmoManager.rotationGizmoEnabled = false;
    gizmoManager.scaleGizmoEnabled = false;
    gizmoManager.boundingBoxGizmoEnabled = false;

    return gizmoManager;
  }

  private _placeCursor() {
    const pickResult = this._scene.pick(
      this._scene.pointerX,
      this._scene.pointerY
    );
    if (pickResult.hit && pickResult.pickedPoint) {
      this.cursorPosition.copyFrom(pickResult.pickedPoint);
      this._cursor.position.copyFrom(this.cursorPosition);
      this._cursor.isVisible = true;
      this.selectObject(this._cursor);
    }
  }

  public selectObject(mesh: AbstractMesh) {
    if (mesh !== this._cursor) {
      this._cursor.isVisible = false;
    }

    if (mesh === this._selectedMesh) {
      return;
    }

    this._selectedMesh = mesh;
    // this._scene.debugLayer.select(selectedMesh);

    this.gizmoManager.positionGizmoEnabled = true;
    this.gizmoManager.rotationGizmoEnabled = false;
    this.gizmoManager.scaleGizmoEnabled = false;
    this.gizmoManager.boundingBoxGizmoEnabled = false;
    this.gizmoManager.attachToMesh(this._selectedMesh);
  }

  public deselectAll(): void {
    this._selectedMesh = null;
    this.gizmoManager.attachToMesh(null);
    this.gizmoManager.positionGizmoEnabled = false;
    this.gizmoManager.rotationGizmoEnabled = false;
    this.gizmoManager.scaleGizmoEnabled = false;
    this.gizmoManager.boundingBoxGizmoEnabled = false;
    this._cursor.isVisible = false;
    this._contextMenu.hide();
  }

  public addPrimitive(
    type: "box" | "sphere" | "cylinder" | "torus",
    position: Vector3,
    rotation: Vector3,
    size: Vector3
  ) {
    let mesh: Mesh;
    switch (type) {
      case "box":
        mesh = MeshBuilder.CreateBox("box", { size: 1 }, this._scene);
        break;
      case "sphere":
        mesh = MeshBuilder.CreateSphere("sphere", { diameter: 1 }, this._scene);
        break;
      case "cylinder":
        mesh = MeshBuilder.CreateCylinder(
          "cylinder",
          { diameterTop: 1, diameterBottom: 1, height: 1 },
          this._scene
        );
        break;
      case "torus":
        mesh = MeshBuilder.CreateTorus(
          "torus",
          { diameter: 1, thickness: 0.5 },
          this._scene
        );
        break;
      default:
        throw new Error("Invalid primitive type");
    }

    mesh.position.copyFrom(position);
    mesh.rotation.copyFrom(rotation);
    mesh.scaling.copyFrom(size);
    mesh.material = this._defaultMaterial;

    this.selectObject(mesh);
  }

  public enableXR() {
    const floor = this._scene.getMeshByName("floor");
    if (!floor) {
      throw new Error("Floor not found");
    }
    this._scene.createDefaultXRExperienceAsync({
      floorMeshes: [floor],
    });
  }
}
