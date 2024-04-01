const content = document.getElementById("content");
const apps = new Map();
const iframes = new Map();
let activeApp = null;
start();

async function loadApp(name) {
  console.log("Loading app: " + name);
  const metadataResponse = await fetch(
    "./apps/" + name + "/metadata.json"
  ).catch((err) => {
    console.error(err);
  });
  const metadata = await metadataResponse.json();

  const url = "./apps/" + name + "/" + metadata.index;

  const iframe = document.createElement("iframe");
  iframe.className = "phone-app";
  iframe.src = url;
  content.appendChild(iframe);

  apps.set(metadata.name, metadata);
  iframes.set(metadata.name, iframe);

  return iframe;
}

async function openApp(name) {
  if (name == activeApp) return;

  if (activeApp) {
    console.log("Sending task to background:", activeApp);
    const activeFrame = iframes.get(activeApp);
    activeFrame.classList.remove("phone-app-active");
  }

  console.log("Opening app:", name);

  activeApp = name;

  const existing = iframes.get(name);
  if (existing) {
    existing.setAttribute("phone-app-active", "true");
    return existing;
  }
  const iframe = await loadApp(name, true);
  iframe.classList.add("phone-app-active");
}

async function start() {
  //await openApp("home");

  // DEBUG: straight to dialer
  await loadApp("home", false);
  openApp("dialer");

}
