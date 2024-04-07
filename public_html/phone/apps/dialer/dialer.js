const keypad = document.getElementById("keypad");
const callStatus = document.getElementById("call-status");
const callStatusText = document.getElementById("call-status-text");
const callStatusTime = document.getElementById("call-status-time");
const callStatusNumber = document.getElementById("call-status-number");
const callSettingsButton = document.getElementById("call-settings-button");
const callSettings = document.getElementById("call-settings");
const callStreamingCheckbox = document.getElementById("call-streaming");
const callBuffersRange = document.getElementById("call-buffers");
const number = document.getElementById("call-number");
const muteButton = document.getElementById("mute-button");
const endButton = document.getElementById("end-button");

const SPEECH_THRESHOLD = 0.25;

let audioContext = null;
let source = null;
let processor = null;
let buffer = [];
let connected = false;
let callStartTime = 0;
let callTimer = null;
let talking = false;
let silence = 0;
let bufferEndTime = 0;
let ringbackBuffer = null;
let ringSource = null;
let sourceNodes = [];
let silenceBuffer = null;
let silenceNode = null;
let ws = null;
let streaming = callStreamingCheckbox.checked;
let prebuffer = parseInt(callBuffersRange.value);

function fetchAudioBuffer(url) {
  console.log("Fetching audio buffer", url);
  return fetch(url)
    .then((response) => response.arrayBuffer())
    .then((arrayBuffer) => audioContext.decodeAudioData(arrayBuffer));
}

async function ringOutbound() {
  if (ringSource) return;
  ringSource = audioContext.createBufferSource();
  if (!ringbackBuffer) {
    await fetchAudioBuffer("res/ringback.wav").then((audioBuffer) => {
      ringbackBuffer = audioBuffer;
    });
  }
  ringSource.buffer = ringbackBuffer;
  ringSource.loop = true;
  ringSource.connect(audioContext.destination);
  ringSource.start();

  silenceNode = audioContext.createBufferSource();
  silenceNode.buffer = silenceBuffer;
  silenceNode.loop = true;
  silenceNode.connect(audioContext.destination);
  silenceNode.start();
}

function stopRinging() {
  if (ringSource) {
    console.log("Stopping ring");
    ringSource.stop();
    ringSource.disconnect();
    ringSource = null;
  }
}

function initializeAudio() {
  const NOISE_LEVEL = 0.0006;
  if (audioContext == null) {
    audioContext = audioContext || new AudioContext();

    // create silence buffer
    const bufferSize = audioContext.sampleRate * 2;
    silenceBuffer = audioContext.createBuffer(
      1,
      bufferSize,
      audioContext.sampleRate
    );
    const data = silenceBuffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * NOISE_LEVEL * 2 - NOISE_LEVEL;
    }
  }
  if (processor == null) {
    processor = audioContext.createScriptProcessor(1024, 1, 1);
  }
}

callSettingsButton.addEventListener("click", () => {
  callSettings.style.display = "flex";
});

document.body.addEventListener("click", (e) => {
  if (e.target !== callSettingsButton && !callSettings.contains(e.target)) {
    callSettings.style.display = "none";
  }
});

callStreamingCheckbox.addEventListener("change", (e) => {
  sendSettings(e);
});

callBuffersRange.addEventListener("input", (e) => {
  sendSettings(e);
});

function sendSettings() {
  streaming = callStreamingCheckbox.checked;
  prebuffer = parseInt(callBuffersRange.value);
  if (ws) {
    const settings = {
      action: "settings",
      streaming: streaming,
      prebuffer: prebuffer,
    };
    console.log("Sending settings", settings);
    ws.send(JSON.stringify(settings));
  }
}

keypad.addEventListener("pointerdown", (e) => {
  const key = e.target.getAttribute("data-key");
  if (!key) return;

  if (key == "backspace") {
    number.innerText = backspace(number.innerText);
    return;
  } else if (key == "send") {
    if (number.innerText.length < 1) return;
    if (number.innerText.length == 8)
      number.innerText.length = "(555) " + number.innerText;
    if ("wakeLock" in navigator) {
      navigator.wakeLock.request("screen").then(() => {
        console.log("Screen wake lock active");
      });
    }
    initializeAudio();
    startCall(number.innerText);
  } else {
    number.innerText = formatPhoneNumber(number.innerText + key);
  }
});

function formatPhoneNumber(input) {
  // up to 11 characters
  // 800
  // 800-5
  // 800-555
  // 800-5551
  // (800) 555-12
  // (800) 555-1212
  // 80055512123
  // (different rules if it starts with a 1)

  let cleanNumber = input.replace(/\D/g, "");

  if (
    cleanNumber.length > 11 ||
    (!cleanNumber.startsWith("1") && cleanNumber.length > 10)
  ) {
    return cleanNumber;
  }
  if (cleanNumber.startsWith("1")) {
    if (cleanNumber.length < 3) {
      return cleanNumber;
    }
    if (cleanNumber.length < 4) {
      return "1 " + cleanNumber.slice(1);
    }
    if (cleanNumber.length < 5) {
      return "1 (" + cleanNumber.slice(1, 4);
    }
    if (cleanNumber.length < 8) {
      return "1 (" + cleanNumber.slice(1, 4) + ") " + cleanNumber.slice(4);
    }
    if (cleanNumber.length < 12) {
      return (
        "1 (" +
        cleanNumber.slice(1, 4) +
        ") " +
        cleanNumber.slice(4, 7) +
        "-" +
        cleanNumber.slice(7)
      );
    }
  } else {
    if (cleanNumber.length < 4) {
      return cleanNumber;
    }
    if (cleanNumber.length < 8) {
      return cleanNumber.slice(0, 3) + "-" + cleanNumber.slice(3);
    }
    if (cleanNumber.length < 11) {
      return `(${cleanNumber.slice(0, 3)}) ${cleanNumber.slice(
        3,
        6
      )}-${cleanNumber.slice(6)}`;
    }
  }
}

function backspace(formattedNumber) {
  const cleanNumber = formattedNumber.replace(/\D/g, "");
  const truncatedNumber = cleanNumber.slice(0, -1);
  return formatPhoneNumber(truncatedNumber);
}

function startCallTimer() {
  callStartTime = new Date().getTime();
  keypad.style.display = "none";
  callStatus.style.display = "inline";

  callTimer = setInterval(() => {
    const time = new Date().getTime() - callStartTime;
    const minutes = Math.floor(time / 60000);
    const seconds = ((time % 60000) / 1000).toFixed(0);
    callStatusTime.innerText = `${minutes}:${
      seconds < 10 ? "0" : ""
    }${seconds}`;
  }, 1000);
}

function showTab(tab) {
  document.querySelectorAll(".tab").forEach((el) => {
    el.style.display = "none";
  });
  document.getElementById(tab).style.display = "flex";
}

async function startCall(phoneNumber) {
  // get mic permissions and add events for audio data
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: false,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  });

  source = audioContext.createMediaStreamSource(stream);

  callStatusText.innerText = "Connecting...";

  processor.onaudioprocess = (event) => {
    if (connected === false) return;

    const input = event.inputBuffer.getChannelData(0);

    // check if we are talking
    if (Math.max(...input) > SPEECH_THRESHOLD) {
      if (!talking) {
        console.log("Speech detected");
        buffer = [];
        for (const source of sourceNodes) {
          source.stop();
        }
        sourceNodes = [];
        ws.send(
          JSON.stringify({
            action: "speech",
            sample_rate: audioContext.sampleRate,
          })
        );
      }
      talking = true;
      silence = 0;
    } else {
      silence++;
      if (talking && silence > 50) {
        talking = false;
        ws.send(
          JSON.stringify({
            action: "pause",
            sample_rate: audioContext.sampleRate,
          })
        );
      }
    }

    if (talking) {
      ws.send(
        JSON.stringify({
          action: "audio",
          sample_rate: audioContext.sampleRate,
        })
      );
      ws.send(input.buffer);
    }

    if (bufferEndTime < audioContext.currentTime) {
      sendBuffers();
    }
  };

  async function sendBuffers() {
    if (buffer.length > 0) {
      const bufferSource = audioContext.createBufferSource();
      const audioBuffer = audioContext.createBuffer(1, buffer.length, 24000);
      audioBuffer.getChannelData(0).set(buffer);
      buffer = [];
      bufferSource.buffer = audioBuffer;
      bufferSource.connect(audioContext.destination);
      if (bufferEndTime < audioContext.currentTime) {
        bufferEndTime = audioContext.currentTime;
      }
      sourceNodes.push(bufferSource);
      bufferSource.start(bufferEndTime);
      bufferEndTime += audioBuffer.duration;
    }
  }

  const target = `wss://${window.location.host}/api/voice/conversation`;
  console.log("Connecting websocket", target);
  ws = new WebSocket(target);
  ws.onopen = () => {
    callStatusNumber.innerText = phoneNumber;
    callStatusText.innerText = "Calling...";
    startCallTimer();
    ws.send(
      JSON.stringify({
        action: "call",
        number: phoneNumber,
        streaming: streaming,
        prebuffer: prebuffer,
      })
    );
  };
  ws.onmessage = (event) => {
    if (event.data instanceof Blob) {
      stopRinging();
      var reader = new FileReader();
      reader.onload = function () {
        const audio_data = new Float32Array(reader.result);
        buffer.push(...audio_data);
      };
      reader.readAsArrayBuffer(event.data);
      return;
    }
    const data = JSON.parse(event.data);
    console.log(data);
    if (data.status == "connected") {
      stopRinging();
      console.log("Call connected");
      connected = true;
      source.connect(processor);
      callStatusText.innerText = "Talking";
      callStatusTime.innerText = "00:00";
      processor.connect(audioContext.destination);
    } else if (data.status == "disconnected") {
      console.log("Call disconnected");
      ws.close();
    } else if (data.status == "error") {
      stopRinging();
      console.log("Call error");
      ws.close();
    } else if (data.status == "busy") {
      stopRinging();
      console.log("Call busy");
      ws.close();
    } else if (data.status == "ringing") {
      callStatusText.innerText = "Ringing...";
      ringOutbound();
    }
  };
  ws.onerror = (error) => {
    console.error("WebSocket Error: ", error);
    stopRinging();
  };
  ws.onclose = (event) => {
    console.log("WebSocket is closed now. Event: ", event);

    connected = false;

    buffer = [];

    stopRinging();
    silenceNode.stop();

    if (connected) {
      processor.disconnect(audioContext.destination);
    }

    stream.getTracks().forEach((track) => track.stop());

    callStatus.style.display = "none";
    keypad.style.display = "inline";

    callStatusTime.innerText = "00:00";
    clearInterval(callTimer);

    if ("wakeLock" in navigator) {
      navigator.wakeLock.release("screen").then(() => {
        console.log("Screen wake lock released");
      });
    }
  };
}
