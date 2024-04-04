const keypad = document.getElementById("keypad");
const callStatus = document.getElementById("call-status-text");
const number = document.getElementById("call-number");
const muteButton = document.getElementById("mute-button");
const endButton = document.getElementById("end-button");

let audioContext = null;
let source = null;
let processor = null;
let buffer = [];
let connected = false;
let callStartTime = 0;
let callStartTimer = null;
let talking = false;
let silence = 0;
let bufferEndTime = 0;

keypad.addEventListener("pointerdown", (e) => {
  const key = e.target.getAttribute("data-key");
  if (!key) return;
  if (key == "backspace") {
    number.innerText = backspace(number.innerText);
    return;
  } else if (key == "send") {
    // set wake lock
    if ("wakeLock" in navigator) {
      navigator.wakeLock.request("screen").then(() => {
        console.log("Screen wake lock active");
      });
    }
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
  callStatus.style.display = "block";

  callStartTimer = setInterval(() => {
    const time = new Date().getTime() - callStartTime;
    const minutes = Math.floor(time / 60000);
    const seconds = ((time % 60000) / 1000).toFixed(0);
    callStatus.innerText = `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`;
  }, 1000);
}

function showTab(tab) {
  document.querySelectorAll(".tab").forEach((el) => {
    el.style.display = "none";
  });
  document.getElementById(tab).style.display = "block";
}

async function startCall(phoneNumber) {
  // get mic permissions and add events for audio data
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: false,
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  });

  audioContext = audioContext || new AudioContext();
  source = audioContext.createMediaStreamSource(stream);
  processor = audioContext.createScriptProcessor(1024, 1, 1);
  source.connect(processor);

  callStatus.innerText = "Connecting...";

  processor.onaudioprocess = (event) => {
    if (connected === false) return;

    const input = event.inputBuffer.getChannelData(0);

    ws.send(
      JSON.stringify({ action: "audio", sample_rate: audioContext.sampleRate })
    );
    ws.send(input.buffer);

    // check if we are talking
    if (Math.max(...input) > 0.25) {
      if (!talking) {
        console.log("Speech detected");
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
      bufferSource.start(bufferEndTime);
      bufferEndTime += audioBuffer.duration;
    }
  }

  const target = `wss://${window.location.host}/api/voice/conversation`;
  console.log("Connecting websocket", target);
  const ws = new WebSocket(target);
  ws.onopen = () => {
    startCallTimer();
    ws.send(JSON.stringify({ action: "call", number: phoneNumber }));
  };
  ws.onmessage = (event) => {
    if (event.data instanceof Blob) {
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
      console.log("Call connected");
      connected = true;
      processor.connect(audioContext.destination);
    } else if (data.status == "disconnected") {
      console.log("Call disconnected");
      ws.close();
    } else if (data.status == "error") {
      console.log("Call error");
      ws.close();
    } else if (data.status == "busy") {
      console.log("Call busy");
      ws.close();
    } else if (data.status == "ringing") {
      callStatus.innerText = "Ringing...";
    }
  };
  ws.onerror = (error) => {
    console.log("WebSocket Error: ", error);
  };
  ws.onclose = (event) => {
    console.log("WebSocket is closed now. Event: ", event);

    connected = false;
    processor.disconnect(audioContext.destination);

    keypad.style.display = "block";
    callStatus.style.display = "none";
    callStatus.innerText = "00:00";
    clearInterval(callStartTimer);

    stream.getTracks().forEach((track) => track.stop());
    if ("wakeLock" in navigator) {
      navigator.wakeLock.release("screen").then(() => {
        console.log("Screen wake lock released");
      });
    }
  };
}
