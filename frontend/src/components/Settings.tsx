import type { UiError } from "../hooks/useVoiceAssistant";

export function VoiceSettings({ voices, voice, onChange }: { voices: string[]; voice: string; onChange: (v: string) => void }) {
  if (voices.length === 0) return null;
  return (
    <section className="panel settings">
      <h2>Voice</h2>
      <label>
        TTS voice{" "}
        <select value={voice} onChange={(e) => onChange(e.target.value)} disabled={voices.length < 2}>
          {voices.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </label>
      <p className="muted small">Takes effect from the next reply. Headphones prevent the assistant from hearing itself.</p>
    </section>
  );
}

export function ErrorBanner({ errors, onDismiss }: { errors: UiError[]; onDismiss: (id: number) => void }) {
  if (errors.length === 0) return null;
  return (
    <div className="errors" role="alert">
      {errors.map((e) => (
        <div key={e.id} className="error">
          <strong>{e.stage}</strong>
          <span>{e.message}</span>
          <button onClick={() => onDismiss(e.id)} aria-label="Dismiss error">
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
