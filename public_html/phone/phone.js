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

class WhisperWebSocketClient {
  constructor(endpoint) {
      this.endpoint = endpoint;
      this.websocket = null;

      this.onConnect = () => {};
      this.onDisconnect = () => {};
      this.onText = (text) => {};
  }

  start() {
      this.websocket = new WebSocket(this.endpoint);

      this.websocket.addEventListener('open', (event) => {
          this.onConnect(event);
      });

      this.websocket.addEventListener('close', (event) => {
          this.onDisconnect(event);
      });

      this.websocket.addEventListener('message', (event) => {
          this.onText(event.data);
      });
  }

  stop() {
      if (this.websocket) {
          this.websocket.close();
      }
  }

  sendAudioChunk(chunk) {
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
          this.websocket.send(chunk);
      }
  }
}

// Example usage:
const whisperClient = new WhisperWebSocketClient('wss://monofy.org/api/whisper/stream');

whisperClient.onConnect = (event) => {
  console.log('Connected to WebSocket');
  // Additional logic on connect if needed
};

whisperClient.onDisconnect = (event) => {
  console.log('Disconnected from WebSocket');
  // Additional logic on disconnect if needed
};

whisperClient.onText = (text) => {
  console.log('Received transcription:', text);
  // Additional logic on receiving text if needed
};

//whisperClient.start();