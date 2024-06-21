export abstract class MathHelpers {
  static linearToGain(linearValue: number): number {
    const clampedValue = Math.max(0, Math.min(1, linearValue));
    const dBValue = clampedValue * 80 - 80;
    const gain = Math.pow(10, dBValue / 20);

    return gain;
  }
}
