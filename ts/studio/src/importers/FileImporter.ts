export abstract class FileImporter {
  static importFile(accept = "*/*"): Promise<File> {
    return new Promise((resolve, reject) => {
      const fileInput = document.createElement("input");
      fileInput.type = "file";
      fileInput.accept = accept;
      fileInput.addEventListener("change", () => {
        const file = fileInput.files?.[0];
        if (file) {
          resolve(file);
        } else {
          reject(new Error("No file selected"));
        }
      });
      fileInput.click();
    });
  }
}
