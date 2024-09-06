export async function checkWebXR() {
  if (!navigator.xr) {
    return new Promise((resolve) => {
      console.log("Loading WebXR polyfill");
      const s = document.createElement("script");
      s.setAttribute("crossorigin", "anonymous");
      s.setAttribute(
        "integrity",
        "sha384-S1Ru9NC0nlEF1j8fc1bS6IjaeFqkB73ib9Yxd3iwkW7Piw5rQRA7i1PY97Y5GIRk"
      );
      s.src =
        "https://cdn.jsdelivr.net/npm/webxr-polyfill@latest/build/webxr-polyfill.js";
      s.onload = resolve;
      document.head.appendChild(s);
    });
  }
}
