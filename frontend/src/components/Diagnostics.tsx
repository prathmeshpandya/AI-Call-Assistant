import type { ConnectionStatus } from "../services/voiceSocket";
import type { ReadyEvent, TurnMetrics } from "../types/protocol";

const fmt = (v: number | null | undefined) => (v === null || v === undefined ? "-" : `${Math.round(v)} ms`);

const ROWS: [keyof TurnMetrics, string][] = [
  ["end_of_speech_to_final_ms", "End of speech -> final transcript"],
  ["asr_finalize_ms", "ASR finalize"],
  ["final_to_first_token_ms", "Final -> first LLM token"],
  ["final_to_first_sentence_ms", "Final -> first sentence"],
  ["final_to_first_audio_ms", "Final -> first audio sent"],
  ["final_to_first_playback_ms", "Final -> playback started"],
  ["end_of_speech_to_first_playback_ms", "End of speech -> playback"],
  ["tts_total_ms", "TTS total"],
  ["response_total_ms", "Response total"],
];

interface Props {
  connection: ConnectionStatus;
  connectionDetail?: string;
  ready: ReadyEvent | null;
  metrics: TurnMetrics[];
  session: { turns: number; interruptions: number; dropped_audio_frames: number };
  rtt: number | null;
}

export function Diagnostics({ connection, connectionDetail, ready, metrics, session, rtt }: Props) {
  const last = metrics[metrics.length - 1];
  const comps = ready?.components;
  return (
    <section className="panel diagnostics">
      <h2>Diagnostics</h2>
      <ul className="health">
        <li className={connection === "open" ? "ok" : "bad"} title={connectionDetail}>
          WebSocket: {connection}
          {rtt !== null && connection === "open" ? ` (${Math.round(rtt)} ms RTT)` : ""}
        </li>
        {comps &&
          (Object.entries(comps) as [string, { ok: boolean; detail: string; engine?: string | null; model?: string }][]).map(
            ([name, c]) => (
              <li key={name} className={c.ok ? "ok" : "bad"} title={c.detail}>
                {name.toUpperCase()}: {c.ok ? c.engine ?? c.model ?? "ok" : "unavailable"}
              </li>
            ),
          )}
      </ul>
      <h3>Last turn {last ? `#${last.turn_id}` : ""}</h3>
      {!last && <p className="muted">Latency appears after the first reply. All values are measured on the server clock.</p>}
      {last && (
        <table>
          <tbody>
            {ROWS.map(([key, label]) => (
              <tr key={key}>
                <th>{label}</th>
                <td>{fmt(last[key] as number | null)}</td>
              </tr>
            ))}
            <tr>
              <th>TTS per sentence</th>
              <td>{last.tts_ms.map((t) => Math.round(t)).join(", ") || "-"}</td>
            </tr>
            <tr>
              <th>Sentences / tokens</th>
              <td>
                {last.sentences} / {last.tokens}
              </td>
            </tr>
            {last.error && (
              <tr className="bad">
                <th>Error</th>
                <td>{last.error}</td>
              </tr>
            )}
          </tbody>
        </table>
      )}
      <p className="muted small">
        Turns: {session.turns} · Interruptions: {session.interruptions} · Dropped mic frames: {session.dropped_audio_frames}
      </p>
    </section>
  );
}
