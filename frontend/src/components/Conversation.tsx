import { useEffect, useRef } from "react";
import type { ChatMessage } from "../hooks/useVoiceAssistant";

export function LiveTranscript({ partial, listening }: { partial: string; listening: boolean }) {
  return (
    <section className="panel live">
      <h2>Live transcription</h2>
      <p className={partial ? "partial" : "partial empty"}>
        {partial || (listening ? "Waiting for speech..." : "Start a conversation to begin.")}
      </p>
    </section>
  );
}

export function ConversationView({ messages }: { messages: ChatMessage[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);
  return (
    <section className="panel conversation">
      <h2>Conversation</h2>
      {messages.length === 0 && <p className="muted">No messages yet.</p>}
      <ol>
        {messages.map((m) => (
          <li key={m.id} className={`msg ${m.role}`}>
            <span className="who">{m.role === "user" ? "You" : "Assistant"}</span>
            <span className="text">
              {m.text || (m.pending ? <em className="muted">thinking...</em> : "")}
              {m.interrupted && <span className="badge">interrupted</span>}
            </span>
          </li>
        ))}
      </ol>
      <div ref={endRef} />
    </section>
  );
}
