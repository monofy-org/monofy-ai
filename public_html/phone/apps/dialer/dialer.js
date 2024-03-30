const keypad = document.getElementById("keypad");
const number = document.getElementById("call-number");
let audioContext = null;
let source = null;
let processor = null;

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
  const ws = new WebSocket(`wss://${window.location.host}/api/conversation`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ call: phoneNumber }));
  };
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
  };
  audioContext = audioContext || new AudioContext();
  source = audioContext.createMediaStreamSource(stream);
  processor = audioContext.createScriptProcessor(1024, 1, 1);
  source.connect(processor);
  processor.connect(audioContext.destination);
  processor.onaudioprocess = (event) => {    
    const data = event.inputBuffer.getChannelData(0);
    const obj = { "action": "audio_chunk", "audio": data.buffer };
    ws.send(JSON.stringify(obj));
  };  
}
