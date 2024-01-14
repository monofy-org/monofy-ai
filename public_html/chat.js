var { hljs, pdfjsLib } = globalThis;
if (pdfjsLib) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.6.347/pdf.worker.entry.min.js";
}
console.log(`PDF support = ${Boolean(pdfjsLib)}`);

const input = document.getElementById("chat-input");
const output = document.getElementById("chat-output");
const speechButton = document.getElementById("speech-button");
const speechPanel = document.getElementById("speech-panel");
const historyButton = document.getElementById("history-button");
const app = document.getElementById("app");
const historyPanel = document.getElementById("history-panel");
const historyList = document.getElementById("history-list");
const ttsMode = document.getElementById("tts-mode");
const localStorageLabel = document.getElementById("local-storage-label");
const localStorageIndicator = document.getElementById("local-storage-indicator");

console.assert(input && output && speechButton && speechPanel && historyButton && historyPanel && historyList);

let currentConversation = null;
let currentConversationButton = null;
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

historyButton.addEventListener("click", () => {
  if (historyPanel.hasAttribute("data-collapsed")) historyPanel.removeAttribute("data-collapsed");
  else historyPanel.setAttribute("data-collapsed", "true");
});

speechButton.addEventListener("click", () => {
  if (speechPanel.hasAttribute("data-collapsed")) speechPanel.removeAttribute("data-collapsed");
  else speechPanel.setAttribute("data-collapsed", "true");
});

async function addMessage(message, bypassMessageLog = false) {

  const isUser = message.role == "user";

  if (!bypassMessageLog) {
    if (output.childNodes.length == 0) {
      startConversation();
    }
    currentConversation.messages.push(message);
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
    if (block.type == "prom") block.type = "prompt"; // these bots be wylin'
    const content = document.createElement("div");
    elt.appendChild(content);
    if (typeof block == "string") {
      content.className = "chat-content";      
      if (!isUser && !bypassMessageLog)
        if (ttsMode.value == "Browser") {
          Speech.speak(block);
        } else if (ttsMode.value == "edge-tts") {
          await fetchAndPlayMP3("./api/tts?model=edge-tts&text=" + encodeURIComponent(block));
        }
        else if (ttsMode.value == "xtts") {
          await fetchAndPlayMP3("./api/tts?model=xtts&text=" + encodeURIComponent(block));
        }
        content.innerText = block;
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
        fetchBase64Image("./api/txt2img?prompt=" + encodeURIComponent(block.text)).then((base64) => {
          imgElement.addEventListener("load", () => {
            output.scrollTop = output.scrollHeight;
            message.images = message.images || [];
            message.images.push(imgElement.src);
            saveMap("conversations", conversations);
          });
          imgElement.src = `data:image/jpeg;base64,${base64}`;
        }).catch((error) => {
          console.error("Error setting image source:", error);
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

  if (!bypassMessageLog && output.childElementCount >= 6 && !currentConversation.title) {
    getSummary(currentConversation);
  }
}

function blankConversation() {
  // display welcome screen
  currentConversation = {
    title: "",
    messages: []
  };
  output.innerHTML = "";
  output.setAttribute("welcome", "true");
}


function startConversation() {
  // first message entered, create new everything
  currentConversation = {
    id: Date.now().toString(),
    title: "",
    messages: [],
  };
  output.innerHTML = "";
  saveConversation(currentConversation.id, currentConversation);
}

function saveConversation(id, conversation) {
  conversations.set(id, conversation);
  // Add the "New Chat" button if this is the first conversation added
  if (historyList.childElementCount == 0) {
    const newButton = document.createElement("div");
    newButton.setAttribute("id", "new-conversation-button");
    newButton.innerText = "New Chat";
    newButton.className = "history-item";
    newButton.addEventListener("click", () => {
      blankConversation();
    });
    historyList.append(newButton);
  }
  // Add conversation to list
  const elt = document.createElement("div");
  currentConversationButton = elt;
  elt.setAttribute("data-conversation", conversation.id);
  elt.setAttribute("id", "conversation-" + conversation.id);
  elt.innerText = conversation.title || "Untitled Chat";
  elt.className = "history-item";
  elt.addEventListener("click", () => {
    currentConversationButton = elt;    
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
    else if (id == currentConversation.id) {
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
    /*
    const opt = {

    };
    */

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

  getCompletion(currentConversation.messages);
}

function loadConversation(conversation) {
  currentConversation = conversation;
  console.log("Loaded conversation:", conversation.title);
  output.innerHTML = "";
  for (const message of conversation.messages) {
    addMessage(message, true);
  }
  output.scrollTop = output.scrollHeight;  
}

function logError(message = "The server is currently down for maintenance. Please try again later.") {
  addMessage({
    role: "system",
    content: message
  }, true);
}

async function getCompletion(messages, bypassMessageLog = false, bypassChatHistory = false) {
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

  return new Promise((resolve, reject) => {
    fetch("./v1/chat/completions", req_settings).then(res => {
      if (!res.ok) {
        logError();
        console.warn(res);
        reject("Request failed.");
      }
      res.json().then(data => {
        if (data.object != "text_completion") {
          console.log("Unexpected response:", data);
          reject("Unexpected response");
        }
        const message = data.choices[0].message;
        if (!bypassChatHistory) addMessage(message, bypassMessageLog);
        resolve(message.content);
      }).catch(error => {
        logError("The server returned an unknown response type. Check console for details.");
        console.warn("The following response was received:", res);
        console.error("Error:", error);
        reject("Error parsing JSON");
      });
    }).catch(error => {
      logError();
      console.error("Error:", error);
      reject(error);
    });
  });
}

function extractContentBetweenBrackets(inputString) {
  const regex = /\{([^}]+)\}/;
  const match = inputString.match(regex);

  if (match) {
    return match[0]; // The entire matched substring, including the brackets
  } else {
    return null; // Return null if no match is found
  }
}

function getSummary(conversation) {  
  const messages = conversation.messages;
  const elementId = "conversation-" + conversation.id;
  const elt = document.getElementById(elementId);
  if (!elt) {
    console.error("Element not found:", elementId);
    return;
  }
  let dump = "";
  for (const message of messages) {
    let prefix = "";
    if (message.role == "user") prefix = "The user said the following: ";
    else if (message.role == "assistant") prefix = "The AI assistant said the following: ";
    else prefix = "";

    dump += `\n\n${prefix}${message.content}\n\nGive your summary in 5 words or less in JSON format.`;
  }
  const content = `You are SummaryBot. Your job is to write witty 5-word (max) tag lines for conversations. You do not generate conversations. You only give a single answer as yourself. Your output is in JSON format like { "summary": "<summary>" } where <summary> represents a 3-4 word tag-line for a given conversation. It should try to encapsulate the overall purpose of the conversation without using more than 3-4 words.\n\nAssistant: Sure, I can do that.\n\nSystem: \n\nHere is the conversation. Afterwards I will hand you a cookie and then you must await your next instructions:\n\n${dump}\n\nDo this right and you get the cookie. Please give a single summary now, and don't forget JSON formatting and 3-5 words max. When done enjoy a cookie and await your next task silently.`;  
  getCompletion(
    [{
      role: "user",
      content: content
    }],
    true,
    true
  ).then(rawText => {
    const text = extractContentBetweenBrackets(rawText);
    const obj = JSON.parse(text);
    if (obj) {
      currentConversation.title = obj.summary;            
      currentConversationButton.innerHTML = currentConversationButton.innerHTML.replace("Untitled Chat", obj.summary);
      saveMap("conversations", conversations);
      if ("summary" in obj) currentConversation.title = obj.summary;
      else console.warn("No summary in response: ", text);
    }
    else console.warn("No summary in response: ", text);
  });
}

function updateSpaceFree() {
  navigator.storage.estimate().then(e => {
    //console.log(e, (e.usage / e.quota));
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

    return new Promise((resolve) => {      
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
    alert("PDF support is unavailable.");
    return;
  }
  // Read the contents of the PDF file as an ArrayBuffer
  const reader = new FileReader();

  reader.onload = async function () {
    const arrayBuffer = this.result;

    // Load the PDF file using pdf.js
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

    // Extract text from each page
    let pdfText = "";
    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
      const page = await pdf.getPage(pageNumber);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map(item => item.str).join(" ");
      pdfText += pageText + "\n"; // Add newline between pages
    }

    // Now call handleTxt with the extracted text
    handleTxt(new Blob([pdfText], { type: "text/plain" }), file.filename);
  };

  // Read the PDF file as an ArrayBuffer
  reader.readAsArrayBuffer(file);
}


function handleTxt(file, overrideFilename) {

  // Read the contents of the file as text
  const reader = new FileReader();

  reader.onload = function (e) {
    const text = e.target.result;

    console.log("File content:", text); // Log the content to inspect

    // Check if the text is human-readable
    if (/*isHumanReadable(text)*/true) {
      console.log("Treating as text");
      if (text.length > 10000) {
        console.error("File is too large. Maximum allowed size is 5000 characters.");
      } else {
        importText(overrideFilename || file.name, text);
      }
    } else {
      console.error("File content is not human-readable.");
    }
  };

  output.removeAttribute("welcome");

  reader.readAsText(file);
}

function handleUnknown(file) {
  console.log("Unknown file type:", file.name);
  console.log("Attempting to treat as text.");
  return handleTxt(file);
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
    const nonPrintableChar = nonPrintableCharMatch ? nonPrintableCharMatch[0] : "unknown";
    const charCode = nonPrintableChar.charCodeAt(0);
    console.error(`Non-printable character found: ${nonPrintableChar} (Char code: ${charCode})`);
    return false;
  }

  return true;
}

function importText(filename, text) {
  console.log("TODO importText:", text);
  sendChat(`\n\n\`\`\`${filename}\n${text}\n\`\`\``);
}

async function fetchBase64Image(url) {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    const base64 = await convertBlobToBase64(blob);
    return base64;
  } catch (error) {
    console.error("Error fetching or converting image:", error);
    throw error;
  }
}

function convertBlobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]);
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