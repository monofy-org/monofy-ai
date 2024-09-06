import { IWorldEngineOptions, WorldEngine } from "../WorldEngine";

export function createDefaultWorld(options: IWorldEngineOptions = {}) {
  const splashScreens = [
    {
      image: "images/splash1.png",
      duration: 2,
    },
    {
      image: "images/splash2.png",
      duration: 2,
    },
  ];

  const terrainHeightmap =
    "https://upload.wikimedia.org/wikipedia/commons/c/c8/Hand_made_terrain_heightmap.png";

  const world = new WorldEngine({
    parentElement: document.body,
    scale: 1024,
    // terrainHeightmap,
    splashScreens,
    ...options,
  });

  world.start();

  return world;
}
