const keypad = document.getElementById("keypad");
const number = document.getElementById("call-number");
let audioContext = null;
let source = null;
let processor = null;
let connected = false;


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
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  audioContext = audioContext || new AudioContext();
  source = audioContext.createMediaStreamSource(stream);
  processor = audioContext.createScriptProcessor(1024, 1, 1);
  source.connect(processor);
  
  const buffer = new Float32Array(1024);
  pos = 0;

  processor.onaudioprocess = (event) => {
    if (connected === false) return;

    const input = event.inputBuffer.getChannelData(0);
    const output = event.outputBuffer.getChannelData(0);
    for (let i = 0; i < input.length; i++) {
      buffer[pos] = input[i];
      pos++;
      if (pos === buffer.length) {
        const data = Array.from(buffer);
        ws.send(JSON.stringify({ action: "audio", data: data }));
        pos = 0;
      }
      output[i] = input[i];
    }
  }  

  const target = `wss://${window.location.host}/api/voice/conversation`;
  console.log("Connecting websocket", target);
  const ws = new WebSocket(target);
  ws.onopen = () => {
    connected = true
    ws.send(JSON.stringify({ action: "call", number: phoneNumber }));
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
    if (data.status == "connected") {
      console.log("Call connected");
      processor.connect(audioContext.destination);
    }
  };
  ws.onerror = (error) => {
    console.log('WebSocket Error: ', error);
  };  
  ws.onclose = (event) => {
    connected = false
    console.log('WebSocket is closed now. Event: ', event);
  };
  
}
