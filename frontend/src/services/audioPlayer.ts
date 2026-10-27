// Gapless, ordered playback of assistant sentences with instant cancellation.
//
// Each sentence arrives as one AudioChunk. Chunks are scheduled back-to-back on the AudioContext
// clock (no gaps, no overlap). Chunks from any generation other than the current one are dropped,
// so audio from an interrupted response can never play.

import type { AudioChunk } from "../types/protocol";

export interface PlayerCallbacks {
  onStarted: (generationId: number, seq: number) => void;
  onIdle: (generationId: number) => void;
  onPlayingChange: (playing: boolean) => void;
}

const SCHEDULE_AHEAD_S = 0.03; // small lead so the first chunk never starts in the past

export class AudioPlayer {
  private ctx: AudioContext | null = null;
  private generation = -1;
  private nextStart = 0;
  private active = new Set<AudioBufferSourceNode>();
  private timers = new Set<number>();
  private ended = new Set<number>(); // generations whose last chunk has been received
  private pendingSeqs = new Map<number, number>(); // highest expected seq per generation
  private expectedNextSeq = 0;
  private buffered = new Map<number, AudioChunk>();

  constructor(private cb: PlayerCallbacks) {}

  /** Must be called from a user gesture (autoplay policy). */
  async init(): Promise<void> {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") await this.ctx.resume();
  }

  beginGeneration(generationId: number): void {
    this.stop();
    this.generation = generationId;
    this.expectedNextSeq = 0;
  }

  enqueue(chunk: AudioChunk): void {
    if (!this.ctx || chunk.generationId !== this.generation) return; // stale audio guard
    // Defensive re-ordering: the server sends in order, but never play out of order.
    this.buffered.set(chunk.seq, chunk);
    while (this.buffered.has(this.expectedNextSeq)) {
      const next = this.buffered.get(this.expectedNextSeq)!;
      this.buffered.delete(this.expectedNextSeq);
      this.expectedNextSeq++;
      this.schedule(next);
    }
  }

  /** The server finished sending this generation (sentences = number of chunks). */
  markEnd(generationId: number, sentences: number): void {
    if (generationId !== this.generation) return;
    this.ended.add(generationId);
    this.pendingSeqs.set(generationId, sentences);
    this.maybeIdle();
  }

  stop(): void {
    for (const src of this.active) {
      src.onended = null;
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    this.active.clear();
    this.timers.forEach((t) => window.clearTimeout(t));
    this.timers.clear();
    this.buffered.clear();
    this.nextStart = 0;
    this.generation = -1;
    this.cb.onPlayingChange(false);
  }

  get isPlaying(): boolean {
    return this.active.size > 0;
  }

  async close(): Promise<void> {
    this.stop();
    await this.ctx?.close().catch(() => undefined);
    this.ctx = null;
  }

  private schedule(chunk: AudioChunk): void {
    const ctx = this.ctx!;
    const buffer = ctx.createBuffer(1, chunk.samples.length, chunk.sampleRate);
    buffer.copyToChannel(chunk.samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + SCHEDULE_AHEAD_S, this.nextStart);
    this.nextStart = startAt + buffer.duration;
    const gen = chunk.generationId;
    src.onended = () => {
      this.active.delete(src);
      if (this.active.size === 0) this.cb.onPlayingChange(false);
      this.maybeIdle();
    };
    src.start(startAt);
    this.active.add(src);
    this.cb.onPlayingChange(true);
    const timer = window.setTimeout(() => {
      this.timers.delete(timer);
      if (gen === this.generation) this.cb.onStarted(gen, chunk.seq);
    }, Math.max(0, (startAt - ctx.currentTime) * 1000));
    this.timers.add(timer);
  }

  private maybeIdle(): void {
    const gen = this.generation;
    if (gen < 0 || !this.ended.has(gen) || this.active.size > 0) return;
    if (this.expectedNextSeq < (this.pendingSeqs.get(gen) ?? 0)) return; // more chunks still in flight
    this.ended.delete(gen);
    this.pendingSeqs.delete(gen);
    this.cb.onIdle(gen);
  }
}
