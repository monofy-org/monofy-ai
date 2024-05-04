let audioContext: AudioContext | null = null;

export function getAudioContext() {
  if (!audioContext) {
    const AudioContext =
      window.AudioContext || (window as any).webkitAudioContext;
    audioContext = new AudioContext();

    console.log("Audio context created");

    // force audio context to start
    const buffer = audioContext.createBuffer(1, 1, audioContext.sampleRate);
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.start(0);
  }
  return audioContext;
}
