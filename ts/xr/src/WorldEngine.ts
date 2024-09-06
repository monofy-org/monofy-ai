import { BaseElement } from "../../elements/src/elements/BaseElement";
import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { ISplashScreen } from "./SplashScreen";
import { GroundMesh } from "@babylonjs/core/Meshes/groundMesh";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { DeviceOrientationCamera } from "@babylonjs/core/Cameras/deviceOrientationCamera";
import { WebXRDefaultExperience } from "@babylonjs/core/XR/webXRDefaultExperience";
import { TerrainHelper } from "./helpers/TerrainHelper";

export interface IWorldEngineOptions {
  splashScreens?: ISplashScreen[];
  scale?: number;
  terrainHeightmap?: string;
  parentElement?: HTMLElement;
}

export class WorldEngine extends BaseElement {
  static readonly FPS = 60;
  static readonly MS_PER_FRAME = 1000 / WorldEngine.FPS;

  canvas: HTMLCanvasElement;
  engine: Engine;
  scene: Scene | null = null;
  terrain: GroundMesh | null = null;

  private _onresize: (() => void) | null = null;

  constructor(readonly options: IWorldEngineOptions) {
    super("canvas", "world-engine");

    this.canvas = this.domElement as HTMLCanvasElement;
    this.engine = new Engine(this.canvas, true);

    this.resize();

    if (options.parentElement) {
      options.parentElement.appendChild(this.canvas);
    }
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
    this.engine.resize();
  }

  async start() {
    console.log("Starting world engine");
    this.scene = await this.createScene();
    if (!this.scene) throw new Error("Scene not created");
    if (this.options.terrainHeightmap) {
      TerrainHelper.loadTerrainFromHeightmap(this.scene, this.options.terrainHeightmap, this.options.scale);
    } else {
      TerrainHelper.createDefaultTerrain(this.scene, this.options.scale);
    }
  }

  stop() {
    this.engine.stopRenderLoop();
    if (this._onresize) {
      window.removeEventListener("resize", this._onresize);
    }
  }

  async createScene() {
    console.log("Creating scene");

    const scene = new Scene(this.engine);
    const camera = new DeviceOrientationCamera(
      "DeviceOrientationCamera",
      new Vector3(-30, -30, -30),
      scene
    );
    camera.setTarget(Vector3.Zero());
    camera.attachControl(this.canvas, true);
    new HemisphericLight("light", new Vector3(0, 0, 0), scene);

    if (navigator.xr) {
      console.log("Starting XR");
      WebXRDefaultExperience.CreateAsync(scene)
        .then((xr) => {
          console.log("XR started", xr);
        })
        .catch(() => {
          console.log("XR creation failed, using fallback renderer");
          
          if (navigator.xr) alert("XR failed/in use");

          this.startFallbackRenderLoop(scene);

          this._onresize = () => {
            this.engine.resize();
          };
          window.addEventListener("resize", this._onresize);
        });
    } else {
      console.log("No XR detected, using fallback renderer");
      this.startFallbackRenderLoop(scene);
    }

    return scene;
  }

  startFallbackRenderLoop(scene: Scene) {
    this.engine.runRenderLoop(() => {
      scene.render();
    });
  }
}
