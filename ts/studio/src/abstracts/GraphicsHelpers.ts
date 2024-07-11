import { IEvent, IPattern } from "../schema";

export abstract class GraphicsHelpers {
  static renderWaveform(
    canvas: HTMLCanvasElement,
    buffer: AudioBuffer,
    color: string = "#000000"
  ) {
    const numChannels = buffer.numberOfChannels;    
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

    // Create ImageData object
    const imgData = ctx.createImageData(width, height);
    const pixels = imgData.data;

    // Clear the canvas with a transparent background
    for (let i = 0; i < pixels.length; i += 4) {
      pixels[i] = 0; // R
      pixels[i + 1] = 0; // G
      pixels[i + 2] = 0; // B
      pixels[i + 3] = 0; // A
    }

    // Parse the color to RGB
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);

    // Loop through each channel and draw its waveform
    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = buffer.getChannelData(channel);
      const bufferLength = channelData.length;
      const samplesPerPixel = Math.floor(bufferLength / width);
      const yOffset = (height / numChannels) * channel;
      const yCenter = yOffset + height / numChannels / 2;

      // Plot the waveform for the current channel
      for (let x = 0; x < width; x++) {
        let minSample = 1.0;
        let maxSample = -1.0;

        // Find min and max samples for this pixel
        for (let j = 0; j < samplesPerPixel; j++) {
          const sample = channelData[x * samplesPerPixel + j];
          if (sample < minSample) minSample = sample;
          if (sample > maxSample) maxSample = sample;
        }

        // Draw a vertical line from min to max
        const yMin = Math.floor(
          yCenter + minSample * (height / numChannels / 2)
        );
        const yMax = Math.floor(
          yCenter + maxSample * (height / numChannels / 2)
        );

        for (let y = yMin; y <= yMax; y++) {
          const index = (y * width + x) * 4;
          pixels[index] = r; // R
          pixels[index + 1] = g; // G
          pixels[index + 2] = b; // B
          pixels[index + 3] = 255; // A
        }
      }
    }

    // Update the canvas with the new image data
    ctx.putImageData(imgData, 0, 0);
  }

  static renderSequence(
    canvas: HTMLCanvasElement,
    events: IEvent[],
    color: string,
    beatWidth: number,
    clearFirst = true
  ) {
    // find lowest note
    let lowestNote = 88;
    let highestNote = 0;
    for (const event of events) {
      if (!(event.note || event.note === 0)) {
        console.log("event", event);
        throw new Error("renderToCanvas() No note!");
      }
      if (event.note < lowestNote) lowestNote = event.note;
      if (event.note > highestNote) highestNote = event.note;
    }

    highestNote += 12;
    lowestNote -= 12;

    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

    if (clearFirst) ctx.clearRect(0, 0, canvas.width, canvas.height);

    const paddedScale = Math.min(
      canvas.height / (highestNote - lowestNote + 1),
      2
    );

    for (const note of events) {
      const y = canvas.height - (note.note! - lowestNote + 1) * paddedScale;
      const x = note.start * beatWidth;
      const width = note.duration * beatWidth;
      ctx.fillStyle = color;
      ctx.fillRect(x, y, width, paddedScale);
    }
  }

  static renderPattern(
    canvas: HTMLCanvasElement,
    pattern: IPattern,
    color: string,
    beatWidth: number
  ) {
    const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const concat: IEvent[] = [];

    for (const sequence of pattern.tracks) {
      if (!sequence.events || sequence.events.length === 0) continue;
      concat.push(...sequence.events);
    }

    if (concat.length > 0) {
      GraphicsHelpers.renderSequence(canvas, concat, color, beatWidth, false);
    }
  }
}
