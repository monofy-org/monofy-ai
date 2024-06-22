let audioContext: AudioContext | null = null;

const AudioContext = window.AudioContext || (window as any).webkitAudioContext;

export function getAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext({
      latencyHint: "interactive",
      sampleRate: 44100,
    });
    getAudioDevices().then((devices) => {
      console.log("Audio devices", devices);
    });
    console.log("Audio context created");    

    return audioContext;
  }

  if (audioContext.state === "suspended") {
    audioContext.resume();
  }

  return audioContext;
}

export async function getAudioDevices() {
  if ("setSinkId" in AudioContext.prototype) {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices;
  }
  return null;
}
