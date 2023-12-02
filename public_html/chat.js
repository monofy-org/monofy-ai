var { pdfjsLib } = globalThis;
if (pdfjsLib) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.6.347/pdf.worker.entry.min.js';
}
console.log(`PDF support = ${Boolean(pdfjsLib)}`);

const LLM_USER_NAME = "User";
const LLM_ASSISTANT_NAME = "Assistant";

const input = document.getElementById("chat-input");
const output = document.getElementById("chat-output");
const speechButton = document.getElementById("speech-button");
const speechPanel = document.getElementById("speech-panel");
const historyButton = document.getElementById("history-button");
const main = document.getElementById("main");
const historyPanel = document.getElementById("history-panel");
const historyList = document.getElementById("history-list");
const newChatButton = document.getElementById("new-chat-button");
const ttsMode = document.getElementById("tts-mode");
const localStorageLabel = document.getElementById("local-storage-label");
const localStorageIndicator = document.getElementById("local-storage-indicator");

console.assert(input && output && speechButton && speechPanel && historyButton && historyPanel && historyList);

let conversationId = null;
let messageHistory = [];
let conversations = new Map();
let audioContext = null;
let source = null;

function getAudioContext() {
  if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
  return audioContext;
}

app.addEventListener("dragover", preventDefaults);
app.addEventListener("drop", handleDrop);

window.addEventListener("load", async () => {
  const saved = await getMap("conversations");
  console.log(saved);
  if (saved) {
    conversations = saved;
    console.log(`Loaded ${saved.size} conversations from local storage.`);
    for (const [id, conversation] of saved)
      saveConversation(id, conversation);
  }
  updateSpaceFree();
});

input.addEventListener("keydown", (e) => {
  if (e.key == "Enter") {
    sendChat(input.value);
    input.value = "";
    e.preventDefault();
  }
});

historyButton.addEventListener("click", (e) => {
  if (historyPanel.hasAttribute("data-collapsed")) historyPanel.removeAttribute("data-collapsed");
  else historyPanel.setAttribute("data-collapsed", "true");
});

speechButton.addEventListener("click", (e) => {
  if (speechPanel.hasAttribute("data-collapsed")) speechPanel.removeAttribute("data-collapsed");
  else speechPanel.setAttribute("data-collapsed", "true");
});

async function addMessage(message, bypassMessageLog = false) {

  const isUser = message.role == "user";

  if (!bypassMessageLog) {
    if (output.childNodes.length == 0) {
      startConversation();
    }
    messageHistory.push(message);
    saveMap("conversations", conversations);
  }

  const elt = document.createElement("div");
  const header = document.createElement("div");
  const avatar = document.createElement("div");
  const name = document.createElement("div");

  header.appendChild(avatar);
  header.appendChild(name);
  elt.appendChild(header);

  elt.className = "chat-message";
  header.className = "chat-message-header";
  avatar.className = "chat-message-avatar";
  name.className = "chat-message-sender";
  name.innerText = message.role;

  output.appendChild(elt);

  const blocks = getCodeBlocks(message.content);

  let imageIndex = 0;

  for (const block of blocks) {
    const content = document.createElement("div");
    elt.appendChild(content);
    if (typeof block == "string") {
      content.className = "chat-content";
      content.innerText = block;
      if (!isUser && !bypassMessageLog)
        if (ttsMode.value == "Browser") {
          Speech.speak(block);
        } else if (ttsMode.value == "edge-tts") {
          await fetchAndPlayMP3("./api/tts?model=edge-tts&text=" + encodeURIComponent(block));
        }
        else if (ttsMode.value == "xtts") {
          await fetchAndPlayMP3("./api/tts?model=xtts&text=" + encodeURIComponent(block));
        }
    } else {
      content.className = "chat-block";
      const blockHeader = document.createElement("h1");
      const blockContent = document.createElement("pre");
      const blockCode = document.createElement("code");
      blockContent.appendChild(blockCode);
      content.appendChild(blockHeader);
      content.appendChild(blockContent);
      console.log("Creating code block:", block);
      blockHeader.innerHTML = block.type;
      blockCode.innerText = block.text;
      if (block.type == "prompt") {
        const imgElement = new Image();
        blockContent.appendChild(imgElement);
        if (bypassMessageLog) {
          if (message.images) imgElement.src = message.images[imageIndex++]; // TODO increment
          continue; // TODO: save/load old images
        }
        output.scrollTop = output.scrollHeight;
        fetchBase64Image("./api/sd?prompt=" + encodeURIComponent(block.text)).then((base64) => {
          imgElement.addEventListener("load", e => {
            output.scrollTop = output.scrollHeight;
            message.images = message.images || [];
            message.images.push(imgElement.src);
            saveMap("conversations", conversations);
          });
          imgElement.src = `data:image/jpeg;base64,${base64}`;
        }).catch((error) => {
          console.error('Error setting image source:', error);
        });
      } else {
        if (["typescript", "javascript"].includes(block.type)) blockCode.className = "language-" + block.type;
        blockCode.innerHTML = hljs.highlight(block.type, block.text).value;
      }
    }
  }

  saveMap("conversations", conversations);

  output.removeAttribute("welcome");

  if (isUser) elt.setAttribute("data-user", "true");
  output.scrollTop = output.scrollHeight;
  //if (output.childElementCount == 4) getSummary(conversationId);
}

function blankConversation() {
  // display welcome screen
  conversationId = null;
  messageHistory = [];
  output.innerHTML = "";
  output.setAttribute("welcome", "true");
}


function startConversation() {
  // first message entered, create new everything
  conversationId = Date.now().toString();
  messageHistory = [];
  saveConversation(conversationId, messageHistory);
}

function saveConversation(id, conversation) {
  conversations.set(id, conversation);
  // Add the "New Chat" button if this is the first conversation added
  if (historyList.childElementCount == 0) {
    const newButton = document.createElement("div");
    newButton.setAttribute("id", "new-conversation-button");
    newButton.innerText = "New Chat";
    newButton.className = "history-item";
    newButton.addEventListener("click", e => {
      blankConversation();
    });
    historyList.append(newButton);
  }
  // Add conversation to list
  const elt = document.createElement("div");
  elt.setAttribute("data-conversation", id);
  elt.setAttribute("id", "conversation-" + id);
  elt.innerText = "Untitled Chat";
  elt.className = "history-item";
  elt.addEventListener("click", e => {
    conversationId = id;
    loadConversation(conversation);
  });
  // Add delete button
  const del = document.createElement("div");
  del.className = "history-item-delete";
  del.innerText = "X";
  del.addEventListener("click", e => {
    e.stopPropagation();
    conversations.delete(id);
    saveMap("conversations", conversations);
    //const fallback = elt.nextElementSibling || elt.previousElementSibling
    //console.log("fallback =", fallback);
    elt.remove();
    if (historyList.childElementCount == 1) {
      // only the "New Chat" button is left
      historyList.innerHTML = "";
      blankConversation();
    }
    else if (id == conversationId) {
      blankConversation();
      /*
      const fallbackId = fallback.getAttribute("data-conversation");
      conversationId = fallbackId;
      console.log("fallbackId =", fallbackId);
      const fallbackConvo = conversations.get(fallbackId);
      console.log(conversations);
      console.log("fallbackConvo =", fallbackConvo);
      loadConversation(fallbackConvo);
      */
    }

    saveMap("conversations", conversations);
  });
  elt.appendChild(del);

  historyList.append(elt);
}

async function fetchAndPlayMP3(url, text) {
  const audioContext = getAudioContext();
  if (text) {
    // TODO: post
    const opt = {

    }

  } else return await fetch(url)
    .then(response => response.arrayBuffer())
    .then(buffer => {
      audioContext.decodeAudioData(buffer, decodedData => {
        if (source) source.stop();
        source = audioContext.createBufferSource();
        source.buffer = decodedData;
        source.connect(audioContext.destination);
        source.start();
      });
    })
    .catch(error => console.error("Error fetching MP3:", error));
}

async function sendChat(text) {

  addMessage({
    role: "user",
    content: text
  });

  getCompletion(messageHistory);
}

function loadConversation(messages) {
  console.log("Loaded conversation:", messages);
  output.innerHTML = "";
  for (const message of messages) {
    addMessage(message, true);
  }
  output.scrollTop = output.scrollHeight;
}

function logError(message = "The server is currently down for maintenance. Please try again later.") {
  addMessage({
    role: "system",
    content: message
  }, true)
}

async function getCompletion(messages, bypassMessageLog) {
  const req = {
    model: "gpt-4",
    messages: messages,
    max_tokens: 500
  };

  const req_settings = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(req)
  };

  return fetch("./v1/chat/completions", req_settings).then(res => {
    if (!res.ok) {
      logError();
      return console.error("Request failed.", res);
    }
    res.json().then(data => {
      if (data.object != "text_completion") {
        return console.error("Unexpected response:", data);
      }
      const message = data.choices[0].message;
      addMessage(message, bypassMessageLog);
    }).catch(error => {
      logError("The server returned an unknown response type. Check console for details.");
      console.warn("The following response was received:", res)
      console.error("Error:", error);
    });
  }).catch(error => {
    logError();
    console.error("Error:", error);
  });
}

function getSummary(id) {
  console.log("Getting summary for conversaionId", id);
  const messages = conversations.get(id);
  if (!messages) {
    console.error("Conversation not found:", id);
    return;
  }
  const elementId = "conversation-" + id;
  const elt = document.getElementById(elementId);
  if (!elt) {
    console.error("Element not found:", elementId);
    return;
  }
  let dump = "";
  for (const message of messages) {

    if (message.role == "user") prefix = LLM_USER_NAME + ": "
    else if (message.role == "assistant") prefix = LLM_ASSISTANT_NAME + ": "
    else prefix = ""

    dump += `\n\n${prefix}${message.content}\n\nGive your summary in 5 words or less in JSON format.`
  }
  content = `You are programmed to generate valid JSON only.\nOutput an object with a property called summary containing a 3-4 word tag-line for a given conversation which try to encapsulate the overall purpose of the conversation without using more than 3-4 words.\nIt is critical that you only output valid JSON and not regular text responses, backquotes, or any other such markup.\nHere is the conversation:\n\n${dump}\n\n\`\`\`json\n`;
  console.log(content);
  getCompletion(
    [{
      role: LLM_USER_NAME,
      content: content
    }],
    true
  ).then(e => {
    console.warn("TODO", e);
  }).catch(e => {
    console.error("Unable to get generate summary:", e);
  });
}

function updateSpaceFree() {
  navigator.storage.estimate().then(e => {
    console.log(e, (e.usage / e.quota));
    const width = ((e.usage / e.quota) * 100).toFixed(2) + "%";
    localStorageLabel.innerText = "Local storage used: " + width;
    localStorageIndicator.style.width = width;
  });
}

class Speech {
  static stop() {
    window.speechSynthesis.cancel();
  }
  static voices = [];

  static initialize() {
    if (this.voices.length > 0) {
      return this.voices;
    }

    window.speechSynthesis.addEventListener("voiceschanged", (e) => {
      console.log("voiceschanged", e);
    });

    return new Promise((resolve, reject) => {
      const synth = window.speechSynthesis;
      const id = setInterval(async () => {
        const v = window.speechSynthesis.getVoices();
        if (v.length > this.voices.length) {
          this.voices = v;
          console.log("voices", this.voices);
          clearInterval(id);
          resolve(v);
        } else {
          // hope voiceschanged fires
        }
      }, 10);
    });
  }

  static async speak(
    text,
    voiceNumber,
    rate = 1,
  ) {
    if (!text) {
      console.warn("Empty string sent to speak()");
      return;
    }
    const phonetic = text
      .replace(/,/g, "")
      .replace(/\-/, " ")
      .replace(/\*[^*]+\*/g, "");
    const utterance = new SpeechSynthesisUtterance(phonetic);
    if (voiceNumber > -1) utterance.voice = this.voices[voiceNumber];
    if (utterance.voice) utterance.lang = utterance.voice.lang;
    else utterance.lang = "en-US";
    utterance.rate = rate;
    window.speechSynthesis.speak(utterance);
    if ("setWakeLock" in window) window.setWakeLock(true);
    return new Promise((resolve) => {
      utterance.onend = () => {
        if ("setWakeLock" in window) window.setWakeLock(true);
        resolve();
      };
    });
  }
}

function getCodeBlocks(text) {
  const textAndCodeBlocks = [];
  const codeBlockPattern = /```(\w+)\s*([\s\S]*?)```/g;

  let lastIndex = 0;
  let match;

  while ((match = codeBlockPattern.exec(text)) !== null) {
    const [, type, code] = match;
    if (lastIndex < match.index) {
      const unblockedText = text.slice(lastIndex, match.index);
      textAndCodeBlocks.push(unblockedText);
    }
    textAndCodeBlocks.push({ type, text: code });
    lastIndex = codeBlockPattern.lastIndex;
  }

  if (lastIndex < text.length) {
    const unblockedText = text.slice(lastIndex);
    textAndCodeBlocks.push(unblockedText);
  } else {
    // If the last code block was not closed, add a closing "```"
    const lastItem = textAndCodeBlocks[textAndCodeBlocks.length - 1];
    if (
      typeof lastItem === "string" &&
      lastItem.startsWith("```") &&
      !lastItem.endsWith("```")
    ) {
      textAndCodeBlocks.pop(); // Remove the incomplete code block
      textAndCodeBlocks.push(lastItem + "```"); // Add the complete code block
      console.warn("Automatically closing unclosed ``` block.");
    }
  }

  return textAndCodeBlocks;
}

function handleFile(file) {
  // Determine file type and call the appropriate handler
  const fileType = file.type;
  if (fileType.startsWith("audio/")) {
    handleAudio(file);
  } else if (fileType.startsWith("image/")) {
    handleImage(file);
  } else if (fileType === "application/pdf") {
    handlePdf(file);
  } else if (fileType === "text/plain") {
    handleTxt(file);
  } else {
    handleUnknown(file);
  }
}

function handleAudio(file) {
  console.log("Handling audio file:", file.name);
  // Implement your logic for handling audio files here
}

function handleImage(file) {
  console.log("Handling image file:", file.name);
  // Implement your logic for handling image files here
}

function handlePdf(file) {
  if (!pdfjsLib) {
    alert("PDF support is unavailable on localhost due to CORS restrictions.");
    return;
  }
  // Read the contents of the PDF file as an ArrayBuffer
  const reader = new FileReader();

  reader.onload = async function () {
    const arrayBuffer = this.result;

    // Load the PDF file using pdf.js
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    // Extract text from each page
    let pdfText = '';
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
      const page = await pdf.getPage(pageNumber);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map(item => item.str).join(' ');
      pdfText += pageText + '\n'; // Add newline between pages
    }

    // Now call handleTxt with the extracted text
    handleTxt(new Blob([pdfText], { type: 'text/plain' }), file.filename);
  };

  // Read the PDF file as an ArrayBuffer
  reader.readAsArrayBuffer(file);
}


function handleTxt(file, overrideFilename) {

  // Read the contents of the file as text
  const reader = new FileReader();

  reader.onload = function (e) {
    const text = e.target.result;

    console.log('File content:', text); // Log the content to inspect

    // Check if the text is human-readable
    if (/*isHumanReadable(text)*/true) {
      console.log('Treating as text');
      if (text.length > 10000) {
        console.error('File is too large. Maximum allowed size is 5000 characters.');
      } else {
        importText(overrideFilename || file.name, text);
      }
    } else {
      console.error('File content is not human-readable.');
    }
  };

  output.removeAttribute("welcome");

  reader.readAsText(file);
}

function handleUnknown(file) {
  console.log("Unknown file type:", file.name);
  console.log("Attempting to treat as text.");
  return handleTxt(file)
}

function preventDefaults(e) {
  e.preventDefault();
  e.stopPropagation();
}

function handleDrop(e) {
  preventDefaults(e);

  const files = e.dataTransfer.files;
  for (const file of files) {
    handleFile(file);
  }
}

function isHumanReadable(text) {
  // Check if the text contains only printable ASCII characters, excluding newline and carriage return
  const printableAsciiRegex = /^[\x20-\x7E\x09\x0A\x0D]*$/;

  if (!printableAsciiRegex.test(text)) {
    const nonPrintableCharMatch = text.match(/[^\x20-\x7E\x09\x0A\x0D]/);
    const nonPrintableChar = nonPrintableCharMatch ? nonPrintableCharMatch[0] : 'unknown';
    const charCode = nonPrintableChar.charCodeAt(0);
    console.error(`Non-printable character found: ${nonPrintableChar} (Char code: ${charCode})`);
    return false;
  }

  return true;
}

function importText(filename, text) {
  console.log('TODO importText:', text);
  sendChat(`\n\n\`\`\`${filename}\n${text}\n\`\`\``);
}

async function fetchBase64Image(url) {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    const base64 = await convertBlobToBase64(blob);
    return base64;
  } catch (error) {
    console.error('Error fetching or converting image:', error);
    throw error;
  }
}

function convertBlobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// Open or create a database
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("myDatabase", 1);

    request.onupgradeneeded = function (event) {
      const db = event.target.result;
      db.createObjectStore("myObjectStore", { keyPath: "key" });
    };

    request.onsuccess = function (event) {
      resolve(event.target.result);
    };

    request.onerror = function (event) {
      reject(event.target.error);
    };
  });
}

// Save a Map to IndexedDB
async function saveMap(key, map) {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(["myObjectStore"], "readwrite");
    const objectStore = transaction.objectStore("myObjectStore");

    const data = { key, value: Array.from(map.entries()) };
    objectStore.put(data);

    await new Promise(resolve => transaction.oncomplete = resolve);
    updateSpaceFree();
  } catch (error) {
    console.error("Error saving to IndexedDB:", error);
  }
}

// Get a Map from IndexedDB
async function getMap(key) {
  try {
    const db = await openDatabase();
    const transaction = db.transaction(["myObjectStore"], "readonly");
    const objectStore = transaction.objectStore("myObjectStore");

    const request = objectStore.get(key);

    const result = await new Promise((resolve, reject) => {
      request.onsuccess = function (event) {
        resolve(event.target.result);
      };

      request.onerror = function (event) {
        reject(event.target.error);
      };
    });

    if (result) {
      return new Map(result.value);
    } else {
      return new Map();
    }
  } catch (error) {
    console.error("Error retrieving from IndexedDB:", error);
    return new Map();
  }
}