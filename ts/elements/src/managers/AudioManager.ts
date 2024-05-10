let audioContext: AudioContext | null = null;

const AudioContext = window.AudioContext || (window as any).webkitAudioContext;

export function getAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
    console.log("Audio context created");
    return audioContext;
  }
  return audioContext;
}
