# SLM Voice Assistant

A local-first, real-time voice assistant: you speak, a **small language model running on your own machine** answers, and you hear the reply. Speech starts **as soon as the first sentence is ready**, and you can **interrupt** it mid-sentence.

- **Speech recognition:** streaming [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) Zipformer with live partial transcripts.
- **Turn-taking:** Silero VAD plus a turn detector that tolerates thinking pauses.
- **LLM:** [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (OpenAI-compatible API) running a quantized instruct model.
- **TTS:** Piper (default, fast) or Kokoro-82M (more natural) through sherpa-onnx, streamed sentence by sentence.
- **Stack:** FastAPI WebSocket backend and a React + TypeScript + Vite frontend. Runs on Windows, CPU-only is fine.

> Docs: [architecture and measurements](docs/architecture.md) · [WebSocket protocol](docs/protocol.md) · [negotiation design](backend/app/negotiation/README.md)

## Status

| Area | State |
|---|---|
| Mic capture → 16 kHz PCM → WebSocket | ✅ Implemented. Worklet resampling verified in a browser; **real-mic test pending on your machine** |
| Streaming STT + VAD + end-of-turn | ✅ Verified end-to-end with real models (synthetic speech input) |
| LLM streaming (llama-server + Qwen3-4B) | ✅ Verified end-to-end; cancellation confirmed in the llama-server log |
| Sentence-level TTS streaming | ✅ Verified end-to-end; measured below |
| Barge-in / interruption | ✅ Verified end-to-end with real models: stops playback, keeps only heard text, and the model follows the correction |
| Latency diagnostics | ✅ Per-turn metrics in the UI |
| Negotiation, telephony, persistent memory | 📐 Design only (see docs) |

### Measured latency (Ryzen 5 5500U, CPU only, Qwen3-4B Q4_K_M, Piper TTS)

These come from `tools/e2e_smoke.py`: real VAD, STT, LLM and TTS, with synthetic speech as input.

| | "Capital of France…?" | "Tips for staying focused?" |
|---|---|---|
| End of speech → final transcript | 748 ms | 737 ms |
| Final → first LLM token | 929 ms | 856 ms |
| Final → first sentence ready | 1972 ms | 1159 ms |
| **End of speech → first audio playing** | **3.2 s** | **2.0 s** |
| Gaps between sentences | 0.2 s total | 2.1 s after a one-word "Sure!", then none |

LLM speed on this CPU is about 9–10 tokens/s for generation and about 40 tokens/s for prompt processing. The system prompt is cached after warm-up. The biggest remaining cost is LLM generation; see [docs/architecture.md](docs/architecture.md) for options.

## Requirements

- Windows 10/11, **Python 3.12**, **Node.js 20+**, Git
- About 2 GB of disk for the speech models, plus 1–3 GB for the LLM
- A microphone. **Headphones are recommended**, so the assistant doesn't hear itself and interrupt.
- Chrome or Edge (tested target)

## 1. Get the models (one time)

Model files are **not** in git (see `.gitignore`). Download them from the project root in PowerShell:

```powershell
# Speech models -> .\models
#   streaming ASR ~310 MB, Silero VAD 2 MB, Piper LJSpeech voice ~67 MB (default TTS),
#   Kokoro ~320 MB (optional higher-quality TTS; add -SkipKokoro to skip it)
powershell -ExecutionPolicy Bypass -File scripts\download-models.ps1
```

**LLM:** this downloads a prebuilt llama.cpp (CPU) and a GGUF model into `.\llm`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup-llm.ps1 -Model qwen3-4b
```

| `-Model` | File (Q4_K_M) | Size | License | Notes |
|---|---|---|---|---|
| `qwen3-4b` (default) | Qwen3-4B-Instruct-2507 | ~2.5 GB | Apache-2.0 | Best quality of the three; a non-"thinking" variant, so it doesn't emit reasoning text |
| `qwen2.5-1.5b` | Qwen2.5-1.5B-Instruct | ~1.1 GB | Apache-2.0 | Choose this if replies feel slow on CPU |
| `llama3.2-3b` | Llama-3.2-3B-Instruct | ~2.0 GB | Llama 3.2 Community License | Requires accepting Meta's license terms |

All three repos and licenses were checked on Hugging Face on 2026-10-27. Speed on your CPU hasn't been measured yet.

## 2. Start the LLM server (terminal 1)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start-llm.ps1
```

This serves the newest `.gguf` in `.\llm` on `http://127.0.0.1:8080` under the alias `local-slm`, which matches `LLM_MODEL`. Check it:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/models
```

## 3. Start the backend (terminal 2)

```powershell
cd backend
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m app.main
```

Health check: `Invoke-RestMethod http://127.0.0.1:8000/api/health`. It reports exactly which component is missing.

## 4. Start the frontend (terminal 3)

```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**, click **Start conversation**, allow the microphone, and speak.

## Configuration

Copy `.env.example` to `.env` in the project root and edit it. Every setting is documented there. These are the ones you're most likely to change:

| Setting | Default | What it does |
|---|---|---|
| `LLM_BASE_URL` / `LLM_MODEL` | `http://127.0.0.1:8080/v1` / `local-slm` | Which OpenAI-compatible server and model to use |
| `TTS_PROVIDER` | `piper_sherpa` | `kokoro_sherpa` (with `TTS_MODEL_DIR=models/kokoro-en-v0_19`) sounds more natural, but on a Ryzen 5 5500U it ran slower than real time while the LLM was generating, causing 4–9 s gaps. Use it on faster hardware |
| `TTS_VOICE` | *(empty)* | The engine's default voice. Kokoro offers 11 voices (e.g. `af_bella`), selectable in the UI |
| `END_OF_TURN_SILENCE_MS` | `700` | Lower is snappier; higher interrupts you less during pauses |
| `BARGE_IN_MIN_SPEECH_MS` / `BARGE_IN_MIN_WORDS` | `400` / `2` | How much speech counts as a real interruption |
| `MAX_HISTORY_TURNS` | `12` | Conversation context window |
| `backend/prompts/system_prompt.txt` | – | The assistant's personality and speaking style |

## Tests

```powershell
cd backend
.venv\Scripts\python.exe -m pytest -q
```

There are 41 unit and integration tests, all using fake model engines. They cover:
- configuration loading and audio and message validation
- sentence chunking, with abbreviations, decimals, initials and run-ons
- turn detection and barge-in rules
- conversation memory, where only heard sentences are kept
- ordered TTS, and overlap of LLM generation with TTS
- cancellation, stale-audio prevention, and repeated interruptions
- WebSocket connect, disconnect and cleanup, and origin checks
- empty transcripts and noise, missing models, and an unavailable LLM server

**End-to-end with real models** (backend and llama-server running). This speaks a question generated by the local TTS into the backend at real-time pace and prints every event and the latency metrics:

```powershell
cd backend
.venv\Scripts\python.exe tools\e2e_smoke.py --text "What is the capital of France?"
.venv\Scripts\python.exe tools\e2e_smoke.py --text "Tell me about the history of Rome." --barge-in "No wait, tell me about Paris instead."
```

It saves the assistant's reply to `e2e_assistant_reply.wav`. The tool doesn't cover the browser microphone or speaker path, so test that manually in the UI.

## Project layout

```
backend/app/
  main.py              FastAPI app, /api/health, lifespan (model load + warm-up)
  config.py            all settings (env / .env)
  services.py          loads engines once; reports missing models
  api/                 websocket.py (transport + loop), protocol.py (validated messages)
  core/                voice_session.py (orchestrator), audio_pipeline.py, turn_detector.py,
                       barge_in.py, response_streamer.py, sentence_chunker.py,
                       conversation_manager.py, session_manager.py, metrics.py, transport.py
  audio/               audio_utils.py (PCM + frame format), vad.py (Silero)
  speech/ llm/ tts/    *_base.py interface + *_service.py implementation
  negotiation/         design doc for the future bargaining module
backend/prompts/       system_prompt.txt
backend/tests/         pytest suite + fakes
backend/tools/         e2e_smoke.py
frontend/src/          hooks/useVoiceAssistant.ts, services/ (socket, capture, player),
                       components/, types/protocol.ts
frontend/public/       pcm-capture-worklet.js
scripts/               model download + llama-server scripts
```

## Responsible use

This assistant always identifies as an AI and must not be used to impersonate a person. Nothing is recorded or stored to disk. Future telephony and negotiation features require explicit consent, disclosure and user-approved limits; see [docs/architecture.md](docs/architecture.md).
