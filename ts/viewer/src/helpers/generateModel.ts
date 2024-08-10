import { Txt2ModelShapERequest } from "../api";

export async function generateModel(text: string) {
  const request: Txt2ModelShapERequest = {
    prompt: text,
    format: "glb",
    num_inference_steps: 32,
  };
  return new Promise<string>((resolve, reject) => {
    fetch("/api/txt2model/shape", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    })
      .then((response) => {
        response
          .blob()
          .then((blob) => {
            resolve(URL.createObjectURL(blob));
          })
          .catch((error) => {
            reject(error);
          });
      })
      .catch((error) => {
        reject(error);
      });
  });
}
