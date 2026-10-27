import { useVoiceAssistant } from "./hooks/useVoiceAssistant";
import { Controls, StatusIndicator } from "./components/Controls";
import { ConversationView, LiveTranscript } from "./components/Conversation";
import { Diagnostics } from "./components/Diagnostics";
import { ErrorBanner, VoiceSettings } from "./components/Settings";

export default function App() {
  const { state, startMic, stopMic, interrupt, newConversation, setVoice, dismissError } = useVoiceAssistant();
  const assistantBusy = state.state === "thinking" || state.state === "speaking" || state.playing;

  return (
    <div className="app">
      <header>
        <h1>SLM Voice Assistant</h1>
        <StatusIndicator state={state.state} playing={state.playing} />
      </header>
      <ErrorBanner errors={state.errors} onDismiss={dismissError} />
      <main>
        <div className="column main-col">
          <Controls
            micOn={state.micOn}
            micLevel={state.micLevel}
            canStart={state.connection === "open"}
            canInterrupt={assistantBusy}
            onStart={startMic}
            onStop={stopMic}
            onInterrupt={interrupt}
            onReset={newConversation}
          />
          <LiveTranscript partial={state.partial} listening={state.micOn} />
          <ConversationView messages={state.messages} />
        </div>
        <aside className="column side-col">
          <Diagnostics
            connection={state.connection}
            connectionDetail={state.connectionDetail}
            ready={state.ready}
            metrics={state.metrics}
            session={state.session}
            rtt={state.lastRttMs}
          />
          <VoiceSettings voices={state.ready?.voices ?? []} voice={state.voice} onChange={setVoice} />
        </aside>
      </main>
      <footer className="muted small">This assistant is an AI. Speech is processed locally on this computer.</footer>
    </div>
  );
}
