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

const win = window as any;

export class Viewer {
  private _canvas: HTMLCanvasElement;
  private _engine: Engine;
  private _scene: Scene;
  private _camera: ArcRotateCamera;
  private _defaultMaterial;

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

    this._engine = new Engine(this._canvas, true);
    this._scene = new Scene(this._engine);
    this._defaultMaterial = new StandardMaterial(
      "defaultMaterial",
      this._scene
    );
    this._defaultMaterial.diffuseColor = new Color3(0.8, 0.8, 0.8); // Off white color
    this._defaultMaterial.specularColor = new Color3(0, 0, 0); // No shininess
    this._camera = new ArcRotateCamera(
      "camera",
      0,
      0,
      10,
      Vector3.Zero(),
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

    let clickedMesh: Nullable<AbstractMesh> = null;
    let selectedMesh: Nullable<AbstractMesh> = null;
    let clickedTime = 0;

    win.addEventListener("keydown", (event: KeyboardEvent) => {
      if (event.key === "Delete") {
        if (selectedMesh) {
          gizmoManager.attachToMesh(null);
          selectedMesh.dispose();
        }
      } else if (event.key === "e") {
        if (selectedMesh) {
          gizmoManager.rotationGizmoEnabled = true;
          gizmoManager.positionGizmoEnabled = false;
        }
      } else if (event.key === "w") {
        if (selectedMesh) {
          gizmoManager.positionGizmoEnabled = true;
          gizmoManager.rotationGizmoEnabled = false;
          gizmoManager.scaleGizmoEnabled = false;
        }
      } else if (event.key === "q") {
        if (selectedMesh) {
          gizmoManager.positionGizmoEnabled = true;
          gizmoManager.rotationGizmoEnabled = true;
          gizmoManager.scaleGizmoEnabled = false;
        }
      } else if (event.key === "r") {
        if (selectedMesh) {
          gizmoManager.positionGizmoEnabled = false;
          gizmoManager.rotationGizmoEnabled = false;
          gizmoManager.scaleGizmoEnabled = true;
        }
      } else if (event.key == "Escape") {
        selectedMesh = null;
        gizmoManager.attachToMesh(null);
        gizmoManager.positionGizmoEnabled = false;
        gizmoManager.rotationGizmoEnabled = false;
        gizmoManager.scaleGizmoEnabled = false;
      }
    });

    this._scene.onPointerObservable.add((eventData) => {
      if (eventData.type === PointerEventTypes.POINTERDOWN) {
        clickedTime = new Date().getTime();
        if (eventData.pickInfo?.hit) {
          clickedMesh = eventData.pickInfo.pickedMesh;
        }
      } else if (eventData.type === PointerEventTypes.POINTERUP) {
        if (new Date().getTime() - clickedTime > 200) {
          return;
        }
        if (
          eventData.pickInfo?.hit &&
          eventData.pickInfo.pickedMesh === clickedMesh
        ) {
          console.log(eventData.pickInfo.pickedMesh?.name);

          // Store the selected mesh
          selectedMesh = eventData.pickInfo.pickedMesh;

          gizmoManager.positionGizmoEnabled = true;
          gizmoManager.rotationGizmoEnabled = false;
          gizmoManager.scaleGizmoEnabled = false;
          gizmoManager.attachToMesh(selectedMesh);
        } else {
          // Deselect the mesh
          selectedMesh = null;
          gizmoManager.attachToMesh(null);
          gizmoManager.positionGizmoEnabled = false;
          gizmoManager.rotationGizmoEnabled = false;
          gizmoManager.scaleGizmoEnabled = false;
        }
      }
    });

    const gizmoManager = new GizmoManager(this._scene);
    gizmoManager.gizmos.positionGizmo = new PositionGizmo();
    gizmoManager.gizmos.rotationGizmo = new RotationGizmo();
    gizmoManager.gizmos.scaleGizmo = new ScaleGizmo();

    this.createFloor();

    this._engine.runRenderLoop(() => {
      this._scene.render();
    });
  }

  public get domElement() {
    return this._canvas;
  }

  public createFloor() {
    const floor = MeshBuilder.CreateGround(
      "floor",
      { width: 10, height: 10 },
      this._scene
    );
    floor.material = this._defaultMaterial;
  }

  public addBox(position: Vector3, rotation: Vector3, size: Vector3) {
    const box = MeshBuilder.CreateBox("box", { size: 1 }, this._scene);
    box.position = position;
    box.rotation = rotation;
    box.scaling = size;
    box.material = this._defaultMaterial;
  }

  public startXR() {
    const floor = this._scene.getMeshByName("floor");
    if (!floor) {
      throw new Error("Floor not found");
    }
    this._scene.createDefaultXRExperienceAsync({
      floorMeshes: [floor],
    });
  }
}

win.Viewer = Viewer;
