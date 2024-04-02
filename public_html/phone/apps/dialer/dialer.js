const keypad = document.getElementById("keypad");
const number = document.getElementById("call-number");
let audioContext = null;
let source = null;
let processor = null;
let connected = false;
let buffer = [];

keypad.addEventListener("pointerdown", (e) => {
  const key = e.target.getAttribute("data-key");
  if (!key) return;
  if (key == "backspace") {
    number.innerText = backspace(number.innerText);
    return;
  } else if (key == "send") {
    // set wake lock
    navigator.wakeLock.request("screen").then((wakeLock) => {
      console.log("Screen wake lock active");
    });
    startCall(number.innerText);
  } else {
    number.innerText = formatPhoneNumber(number.innerText + key);
  }
});

function formatPhoneNumber(input) {
  const cleanNumber = input.replace(/\D/g, "");

  if (cleanNumber.length > 11) {
    return cleanNumber;
  }

  let formattedNumber = "";

  for (let i = 0; i < cleanNumber.length; i++) {
    if (i === 0 && cleanNumber.length === 11) {
      formattedNumber += cleanNumber[i];
    } else if (i === 1 || i === 4 || i === 7) {
      formattedNumber +=
        (i === 1 ? " (" : i === 4 ? ") " : "-") + cleanNumber[i];
    } else {
      formattedNumber += cleanNumber[i];
    }
  }

  return formattedNumber;
}

function backspace(formattedNumber) {
  const cleanNumber = formattedNumber.replace(/\D/g, "");
  const truncatedNumber = cleanNumber.slice(0, -1);
  return formatPhoneNumber(truncatedNumber);
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

  keypad.style.display = "none";

  let talking = false;
  let silence = 0;
  let bufferEndTime = 0;

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
  };

  const target = `wss://${window.location.host}/api/voice/conversation`;
  console.log("Connecting websocket", target);
  const ws = new WebSocket(target);
  ws.onopen = () => {
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
      processor.disconnect(audioContext.destination);
      ws.close();
    } else if (data.status == "error") {
      console.log("Call error");
      processor.disconnect(audioContext.destination);
      ws.close();
    } else if (data.status == "busy") {
      console.log("Call busy");
      processor.disconnect(audioContext.destination);
      ws.close();
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

    stream.getTracks().forEach((track) => track.stop());
    navigator.wakeLock.release("screen").then(() => {
      console.log("Screen wake lock released");
    });
  };
}
