import type { AssistantState } from "../types/protocol";

const LABELS: Record<AssistantState, string> = {
  idle: "Microphone off",
  listening: "Listening",
  user_speaking: "Hearing you",
  thinking: "Thinking",
  speaking: "Speaking",
};

export function StatusIndicator({ state, playing }: { state: AssistantState; playing: boolean }) {
  return (
    <div className={`status status-${state}`} role="status" aria-live="polite">
      <span className="status-dot" />
      <span>{LABELS[state]}</span>
      {playing && (
        <span className="eq" aria-label="Audio playing">
          <i />
          <i />
          <i />
        </span>
      )}
    </div>
  );
}

interface ControlsProps {
  micOn: boolean;
  micLevel: number;
  canStart: boolean;
  canInterrupt: boolean;
  onStart: () => void;
  onStop: () => void;
  onInterrupt: () => void;
  onReset: () => void;
}

export function Controls(p: ControlsProps) {
  return (
    <div className="controls">
      <button
        className={`mic ${p.micOn ? "on" : ""}`}
        onClick={p.micOn ? p.onStop : p.onStart}
        disabled={!p.micOn && !p.canStart}
        aria-pressed={p.micOn}
        style={{ ["--level" as string]: Math.min(1, p.micLevel * 3).toFixed(2) }}
      >
        <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
          <path
            fill="currentColor"
            d="M12 14a3 3 0 0 0 3-3V5a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2Z"
          />
        </svg>
        <span>{p.micOn ? "End conversation" : "Start conversation"}</span>
      </button>
      <button className="secondary" onClick={p.onInterrupt} disabled={!p.canInterrupt}>
        Interrupt
      </button>
      <button className="secondary" onClick={p.onReset}>
        New conversation
      </button>
    </div>
  );
}
