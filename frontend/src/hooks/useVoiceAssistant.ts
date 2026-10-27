// Wires socket, microphone and player together and exposes UI state.

import { useCallback, useEffect, useReducer, useRef } from "react";
import { AudioCapture, MicPermissionError } from "../services/audioCapture";
import { AudioPlayer } from "../services/audioPlayer";
import { VoiceSocket, type ConnectionStatus } from "../services/voiceSocket";
import type { AssistantState, ReadyEvent, ServerEvent, TurnMetrics } from "../types/protocol";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  interrupted?: boolean;
  pending?: boolean;
  generationId?: number;
}

export interface UiError {
  id: number;
  stage: string;
  code: string;
  message: string;
}

export interface VoiceUiState {
  connection: ConnectionStatus;
  connectionDetail?: string;
  ready: ReadyEvent | null;
  state: AssistantState;
  micOn: boolean;
  micLevel: number;
  partial: string;
  messages: ChatMessage[];
  metrics: TurnMetrics[];
  session: { turns: number; interruptions: number; dropped_audio_frames: number };
  voice: string;
  playing: boolean;
  errors: UiError[];
  lastRttMs: number | null;
}

type Action =
  | { type: "connection"; status: ConnectionStatus; detail?: string }
  | { type: "server"; ev: ServerEvent }
  | { type: "mic"; on: boolean }
  | { type: "level"; level: number }
  | { type: "playing"; playing: boolean }
  | { type: "error"; stage: string; code: string; message: string }
  | { type: "dismiss"; id: number }
  | { type: "rtt"; ms: number }
  | { type: "clear" };

const initial: VoiceUiState = {
  connection: "connecting",
  ready: null,
  state: "idle",
  micOn: false,
  micLevel: 0,
  partial: "",
  messages: [],
  metrics: [],
  session: { turns: 0, interruptions: 0, dropped_audio_frames: 0 },
  voice: "",
  playing: false,
  errors: [],
  lastRttMs: null,
};

let errorSeq = 0;

function reducer(s: VoiceUiState, a: Action): VoiceUiState {
  switch (a.type) {
    case "connection":
      return { ...s, connection: a.status, connectionDetail: a.detail, ...(a.status !== "open" ? { micOn: false, state: "idle" as const } : {}) };
    case "mic":
      return { ...s, micOn: a.on, micLevel: a.on ? s.micLevel : 0 };
    case "level":
      return { ...s, micLevel: a.level };
    case "playing":
      return { ...s, playing: a.playing };
    case "rtt":
      return { ...s, lastRttMs: a.ms };
    case "error":
      return { ...s, errors: [...s.errors.slice(-4), { id: ++errorSeq, stage: a.stage, code: a.code, message: a.message }] };
    case "dismiss":
      return { ...s, errors: s.errors.filter((e) => e.id !== a.id) };
    case "clear":
      return { ...s, messages: [], partial: "", metrics: [] };
    case "server":
      return applyServerEvent(s, a.ev);
  }
}

function applyServerEvent(s: VoiceUiState, ev: ServerEvent): VoiceUiState {
  switch (ev.type) {
    case "ready":
      return { ...s, ready: ev, voice: ev.voice };
    case "state":
      return { ...s, state: ev.state };
    case "transcript":
      if (!ev.final) return { ...s, partial: ev.text };
      {
        // Consecutive user turns are merged server-side; mirror that so the UI matches the model's context.
        const last = s.messages[s.messages.length - 1];
        const messages =
          last && last.role === "user"
            ? [...s.messages.slice(0, -1), { ...last, text: ev.user_turn_text }]
            : [...s.messages, { id: `u${ev.turn_id}`, role: "user" as const, text: ev.text }];
        return { ...s, partial: "", messages };
      }
    case "transcript_ignored":
    case "vad":
      return ev.type === "vad" && ev.event === "discarded" ? { ...s, partial: "" } : s;
    case "assistant_start":
      return {
        ...s,
        messages: [...s.messages, { id: `a${ev.generation_id}`, role: "assistant", text: "", pending: true, generationId: ev.generation_id }],
      };
    case "assistant_text":
      return {
        ...s,
        messages: s.messages.map((m) =>
          m.generationId === ev.generation_id ? { ...m, text: m.text ? `${m.text} ${ev.text}` : ev.text } : m,
        ),
      };
    case "assistant_final":
      return {
        ...s,
        messages: s.messages
          .map((m) =>
            m.generationId === ev.generation_id
              ? { ...m, text: ev.text, interrupted: ev.interrupted, pending: false }
              : m,
          )
          .filter((m) => !(m.generationId === ev.generation_id && !m.text)), // nothing was heard
      };
    case "metrics":
      return { ...s, metrics: [...s.metrics.slice(-19), ev.turn], session: ev.session };
    case "voice_changed":
      return { ...s, voice: ev.voice };
    case "history_reset":
      return { ...s, messages: [], partial: "" };
    case "error":
      return { ...s, errors: [...s.errors.slice(-4), { id: ++errorSeq, stage: ev.stage, code: ev.code, message: ev.message }] };
    default:
      return s;
  }
}

export function useVoiceAssistant() {
  const [state, dispatch] = useReducer(reducer, initial);
  const socketRef = useRef<VoiceSocket | null>(null);
  const captureRef = useRef<AudioCapture | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);

  useEffect(() => {
    const player = new AudioPlayer({
      onStarted: (gen, seq) => socketRef.current?.send({ type: "playback", event: "started", generation_id: gen, seq }),
      onIdle: (gen) => socketRef.current?.send({ type: "playback", event: "idle", generation_id: gen }),
      onPlayingChange: (playing) => dispatch({ type: "playing", playing }),
    });
    playerRef.current = player;

    const socket = new VoiceSocket({
      onStatus: (status, detail) => {
        dispatch({ type: "connection", status, detail });
        if (status !== "open") {
          player.stop();
          void captureRef.current?.stop();
          captureRef.current = null;
        }
      },
      onAudio: (chunk) => player.enqueue(chunk),
      onEvent: (ev) => {
        switch (ev.type) {
          case "assistant_start":
            player.beginGeneration(ev.generation_id);
            break;
          case "assistant_end":
            player.markEnd(ev.generation_id, ev.sentences);
            break;
          case "interrupted":
            player.stop(); // cut audio immediately
            break;
          case "pong":
            if (ev.t) dispatch({ type: "rtt", ms: performance.now() - ev.t });
            break;
        }
        dispatch({ type: "server", ev });
      },
    });
    socketRef.current = socket;
    socket.connect();

    const ping = window.setInterval(() => socket.send({ type: "ping", t: performance.now() }), 5000);
    return () => {
      window.clearInterval(ping);
      socket.close();
      void captureRef.current?.stop();
      void player.close();
    };
  }, []);

  const startMic = useCallback(async () => {
    const socket = socketRef.current;
    if (!socket?.isOpen) {
      dispatch({ type: "error", stage: "connection", code: "not_connected", message: "Not connected to the backend." });
      return;
    }
    try {
      await playerRef.current?.init(); // unlock audio output inside the click gesture
      const capture = new AudioCapture();
      await capture.start({
        onFrame: (pcm) => socket.sendAudio(pcm),
        onLevel: (level) => dispatch({ type: "level", level }),
      });
      captureRef.current = capture;
      socket.send({ type: "start" });
      dispatch({ type: "mic", on: true });
    } catch (err) {
      const message = err instanceof MicPermissionError ? err.message : `Audio setup failed: ${(err as Error).message}`;
      dispatch({ type: "error", stage: "microphone", code: "mic_error", message });
    }
  }, []);

  const stopMic = useCallback(async () => {
    socketRef.current?.send({ type: "stop" });
    playerRef.current?.stop();
    await captureRef.current?.stop();
    captureRef.current = null;
    dispatch({ type: "mic", on: false });
  }, []);

  const interrupt = useCallback(() => {
    playerRef.current?.stop();
    socketRef.current?.send({ type: "interrupt" });
  }, []);

  const newConversation = useCallback(() => {
    playerRef.current?.stop();
    socketRef.current?.send({ type: "reset" });
    dispatch({ type: "clear" });
  }, []);

  const setVoice = useCallback((voice: string) => socketRef.current?.send({ type: "set_voice", voice }), []);
  const dismissError = useCallback((id: number) => dispatch({ type: "dismiss", id }), []);

  return { state, startMic, stopMic, interrupt, newConversation, setVoice, dismissError };
}
