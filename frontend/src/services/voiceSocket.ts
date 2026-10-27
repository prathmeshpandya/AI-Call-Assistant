// WebSocket client with automatic reconnection. JSON text frames = events, binary frames = audio.

import { decodeAudioFrame, type AudioChunk, type ClientMessage, type ServerEvent } from "../types/protocol";

export type ConnectionStatus = "connecting" | "open" | "closed";

export interface SocketCallbacks {
  onEvent: (ev: ServerEvent) => void;
  onAudio: (chunk: AudioChunk) => void;
  onStatus: (status: ConnectionStatus, detail?: string) => void;
}

function defaultUrl(): string {
  const override = import.meta.env.VITE_WS_URL as string | undefined;
  if (override) return override;
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`; // proxied to the backend by Vite in development
}

export class VoiceSocket {
  private ws: WebSocket | null = null;
  private retryMs = 1000;
  private retryTimer: number | null = null;
  private closedByUser = false;

  constructor(private cb: SocketCallbacks, private url: string = defaultUrl()) {}

  connect(): void {
    this.closedByUser = false;
    this.cb.onStatus("connecting");
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retryMs = 1000;
      this.cb.onStatus("open");
    };
    ws.onmessage = (e: MessageEvent) => {
      if (e.data instanceof ArrayBuffer) {
        this.cb.onAudio(decodeAudioFrame(e.data));
      } else {
        try {
          this.cb.onEvent(JSON.parse(e.data as string) as ServerEvent);
        } catch {
          console.warn("Malformed server message", e.data);
        }
      }
    };
    ws.onclose = (e) => {
      this.ws = null;
      if (this.closedByUser) {
        this.cb.onStatus("closed");
        return;
      }
      this.cb.onStatus("closed", `Disconnected (code ${e.code}). Reconnecting in ${this.retryMs / 1000}s...`);
      this.retryTimer = window.setTimeout(() => this.connect(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, 10000);
    };
    this.ws = ws;
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  send(msg: ClientMessage): void {
    if (this.isOpen) this.ws!.send(JSON.stringify(msg));
  }

  sendAudio(pcm: ArrayBuffer): void {
    // Drop audio rather than buffer it while disconnected or congested (stay real-time).
    if (this.isOpen && this.ws!.bufferedAmount < 64_000) this.ws!.send(pcm);
  }

  close(): void {
    this.closedByUser = true;
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer);
    this.ws?.close();
  }
}
