import MidiPlayer from "midi-player-js";
import { IEvent, IPattern, ISequence } from "../schema";

// Function to parse a specific MIDI channel and generate IEvent list
export function parseMidiChannelToEvents(
  midiPlayer: MidiPlayer.Player,
  channelNumber: number,
  bpm: number
): ISequence {
  const events: IEvent[] = [];
  const activeNotes: Record<number, { startTick: number; velocity: number }> =
    {};

  midiPlayer.tracks![channelNumber].events.forEach(
    (event: MidiPlayer.Event) => {
      const { name, tick, velocity, noteNumber, channel } = event;

      // Ensure the event is for the specified channel
      if (channel === channelNumber) {
        // Convert ticks to time based on BPM (assuming 480 ticks per beat)
        const ticksPerBeat = midiPlayer.division; // Use division to get the ticks per beat
        const secondsPerBeat = 60 / bpm;

        if (name === "Note on" && velocity) {
          // Store the note-on event with its start tick and velocity
          activeNotes[noteNumber!] = {
            startTick: tick,
            velocity: velocity!,
          };
        } else if (
          (name === "Note off" || (name === "Note on" && velocity === 0)) &&
          activeNotes[noteNumber!]
        ) {
          // Calculate the duration of the note
          const start =
            (activeNotes[noteNumber!].startTick / ticksPerBeat) *
            secondsPerBeat;
          const duration =
            ((tick - activeNotes[noteNumber!].startTick) / ticksPerBeat) *
            secondsPerBeat;

          // Create IEvent object
          const midiEvent: IEvent = {
            label: "Note",
            start: start,
            duration: duration,
            note: noteNumber!,
            velocity: activeNotes[noteNumber!].velocity,
          };

          events.push(midiEvent);

          // Remove the active note
          delete activeNotes[noteNumber!];
        }
      }
    }
  );

  return { events } as ISequence;
}

export async function parseMidiFileToEventsByChannel(
  midiFile: File,
  bpm: number
): Promise<IPattern> {
  const buffer = await midiFile.arrayBuffer();

  return new Promise((resolve, reject) => {
    try {
      const player = new MidiPlayer.Player(() => {
        /* No event handling in the main player */
      });

      player.loadArrayBuffer(buffer);

      const tracks: ISequence[] = [];
      // Parse each channel to generate IEvent list

      for (let i = 0; i < player.tracks!.length; i++) {
        tracks[i] = parseMidiChannelToEvents(player, i, bpm);
      }

      resolve({ name: midiFile.name, tracks } as IPattern);
    } catch (error) {
      reject(`Error loading MIDI file: ${error}`);
    }
  });
}
