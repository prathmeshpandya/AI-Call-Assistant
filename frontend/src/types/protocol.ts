// WebSocket protocol types. Keep in sync with docs/protocol.md and backend/app/api/protocol.py.

export type AssistantState = "idle" | "listening" | "user_speaking" | "thinking" | "speaking";

export interface ComponentStatus {
  ok: boolean;
  detail: string;
  engine?: string | null;
  model?: string;
}

export interface ReadyEvent {
  type: "ready";
  session_id: string;
  input_sample_rate: number;
  tts_sample_rate: number | null;
  voices: string[];
  voice: string;
  language: string;
  components: { stt: ComponentStatus; tts: ComponentStatus; vad: ComponentStatus; llm: ComponentStatus };
  config: { end_of_turn_silence_ms: number; barge_in_enabled: boolean };
}

export interface TurnMetrics {
  turn_id: number;
  speech_duration_ms: number | null;
  speech_start_to_final_ms: number | null;
  end_of_speech_to_final_ms: number | null;
  asr_finalize_ms: number | null;
  final_to_first_token_ms: number | null;
  final_to_first_sentence_ms: number | null;
  final_to_first_audio_ms: number | null;
  final_to_first_playback_ms: number | null;
  end_of_speech_to_first_playback_ms: number | null;
  tts_ms: number[];
  tts_total_ms: number;
  audio_total_ms: number;
  response_total_ms: number | null;
  sentences: number;
  tokens: number;
  interrupted: boolean;
  error: string | null;
}

export type ServerEvent =
  | ReadyEvent
  | { type: "state"; state: AssistantState }
  | { type: "vad"; event: "speech_start" | "discarded"; reason?: string }
  | { type: "transcript"; final: false; text: string }
  | { type: "transcript"; final: true; text: string; turn_id: number; user_turn_text: string }
  | { type: "transcript_ignored"; text: string; reason: string }
  | { type: "assistant_start"; generation_id: number; turn_id: number }
  | { type: "assistant_text"; generation_id: number; seq: number; text: string }
  | { type: "assistant_end"; generation_id: number; sentences: number }
  | { type: "assistant_final"; generation_id: number; interrupted: boolean; text: string }
  | { type: "interrupted"; generation_id: number; reason: string }
  | { type: "metrics"; turn: TurnMetrics; session: { turns: number; interruptions: number; dropped_audio_frames: number } }
  | { type: "voice_changed"; voice: string }
  | { type: "history_reset" }
  | { type: "pong"; t: number | null }
  | { type: "error"; code: string; message: string; stage: string };

export type ClientMessage =
  | { type: "start" }
  | { type: "stop" }
  | { type: "interrupt" }
  | { type: "reset" }
  | { type: "ping"; t: number }
  | { type: "set_voice"; voice: string }
  | { type: "playback"; event: "started"; generation_id: number; seq: number }
  | { type: "playback"; event: "idle"; generation_id: number };

/** Binary assistant audio frame: 12-byte header (uint32 LE generation_id, seq, sample_rate) + PCM16 LE mono. */
export interface AudioChunk {
  generationId: number;
  seq: number;
  sampleRate: number;
  samples: Float32Array;
}

export function decodeAudioFrame(buf: ArrayBuffer): AudioChunk {
  const view = new DataView(buf);
  const generationId = view.getUint32(0, true);
  const seq = view.getUint32(4, true);
  const sampleRate = view.getUint32(8, true);
  const pcm = new Int16Array(buf, 12, (buf.byteLength - 12) / 2);
  const samples = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) samples[i] = pcm[i] / 32768;
  return { generationId, seq, sampleRate, samples };
}
