export function createViewer(canvasElement?: HTMLCanvasElement) {
  return new Promise((resolve, reject) => {
    import("./viewer")
      .then((module) => {
        const viewer = new module.Viewer(canvasElement);
        resolve(viewer);
      })
      .catch((error) => {
        reject(error);
      });
  });
}

