let audioContext: AudioContext | null = null;

export function getAudioContext() {
  if (!audioContext) {
    const AudioContext =
      window.AudioContext || (window as any).webkitAudioContext;
    audioContext = new AudioContext();
  }
  return audioContext;
}
