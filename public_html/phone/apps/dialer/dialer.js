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
  processor = audioContext.createScriptProcessor(2048, 1, 1);
  source.connect(processor);

  let talking = false;
  let silence = 0;

  processor.onaudioprocess = (event) => {
    if (connected === false) return;

    const input = event.inputBuffer.getChannelData(0);
    const output = event.outputBuffer.getChannelData(0);

    const data = new Uint8Array(input.buffer);
    const data_base64 = btoa(String.fromCharCode(...data));
    ws.send(
      JSON.stringify({
        action: "audio",
        sample_rate: audioContext.sampleRate,
        data: data_base64,
      })
    );

    // check if we are talking
    if (Math.max(...input) > 0.2) {
      if (talking === false) {
        console.log("Speech detected");
      }
      talking = true;
      silence = 0;
    } else {
      silence++;
      if (silence > 50 && talking === true) {
        talking = false;
        ws.send(
          JSON.stringify({
            action: "pause",
            sample_rate: audioContext.sampleRate,
          })
        );
      }
    }

    for (let i = 0; i < output.length && buffer.length; i++) {
      output[i] = buffer.shift();
    }

  };

  const target = `wss://${window.location.host}/api/voice/conversation`;
  console.log("Connecting websocket", target);
  const ws = new WebSocket(target);
  ws.onopen = () => {
    connected = true;
    ws.send(JSON.stringify({ action: "call", number: phoneNumber }));
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
    if (data.status == "connected") {
      console.log("Call connected");
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
    } else if ("audio" in data && data.audio) {
      const audio_data = atob(data.audio);
      const audio_array = new Uint8Array(audio_data.length);
      for (let i = 0; i < audio_data.length; i++) {
        audio_array[i] = audio_data.charCodeAt(i);
      }
      buffer.push(...audio_array);
    }
  };
  ws.onerror = (error) => {
    console.log("WebSocket Error: ", error);
  };
  ws.onclose = (event) => {
    connected = false;
    console.log("WebSocket is closed now. Event: ", event);
  };
}
