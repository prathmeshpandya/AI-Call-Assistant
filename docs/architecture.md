# Architecture

```
 Browser (React + TS)                                     Backend (FastAPI, asyncio)
 ┌──────────────────────────────┐   WebSocket /ws   ┌──────────────────────────────────────────────────────────┐
 │ Mic → AudioWorklet           │ ── PCM16 16k ───▶ │ api/websocket.py   (WebSocketTransport, origin check)     │
 │   (48k→16k, 32 ms frames)    │                   │        │                                                  │
 │                              │ ◀── JSON events ─ │ core/session_manager.py ─▶ core/voice_session.py          │
 │ AudioPlayer (gapless queue,  │ ◀── audio frames  │   ├─ audio queue ─▶ core/audio_pipeline.py (worker thread)│
 │   generation guard)          │                   │   │     Silero VAD → TurnDetector → streaming Zipformer   │
 │ useVoiceAssistant (state)    │ ── playback ack ▶ │   ├─ barge_in.py  (is this a real interruption?)          │
 │ Diagnostics / Conversation   │                   │   ├─ conversation_manager.py (heard-only memory)          │
 └──────────────────────────────┘                   │   └─ response_streamer.py                                 │
                                                    │        LLM task ─▶ SentenceChunker ─▶ Queue ─▶ TTS task   │
                                                    │           │                                  │            │
                                                    │   llm/llm_service.py              tts/tts_service.py     │
                                                    └───────────┼──────────────────────────────────┼────────────┘
                                                                ▼                                  ▼
                                                     llama-server (llama.cpp)          Kokoro / Piper (sherpa-onnx)
                                                     OpenAI-compatible, :8080
```

## Key decisions

| Area | Decision | Why |
|---|---|---|
| Audio transport | Raw PCM16 at 16 kHz over WebSocket, 32 ms frames | It's the native format of Silero VAD and the ASR model, so the server never resamples. 32 ms equals one Silero window. Bandwidth is only 32 KB/s on localhost |
| Browser resampling | An AudioWorklet does box-filter decimation | It runs off the main thread, and the result was verified at 16 000 samples/s from a 48 kHz context. A 16 kHz `AudioContext` isn't reliable across browsers |
| STT | sherpa-onnx streaming Zipformer transducer (int8 encoder) | True streaming with live partials and a Windows wheel. Measured at about 0.11× real time with VAD on a Ryzen 5 5500U |
| VAD | Silero via sherpa-onnx | Needs no PyTorch. Turn logic lives in our own `TurnDetector`, which is pure and tested |
| End of turn | 700 ms of silence, or 1500 ms if the partial ends in "and", "so", "because"... | Avoids replying during thinking pauses without slowing normal turns |
| LLM | llama.cpp `llama-server`, OpenAI-compatible SSE | Replaceable through `LLM_BASE_URL` and `LLM_MODEL`. Closing the HTTP stream aborts generation |
| TTS | Piper LJSpeech (default) or Kokoro v0.19 fp32, through sherpa-onnx 1.12 | Kokoro couldn't keep up in real time on this CPU while the LLM was generating (see below). LJSpeech is public-domain data; the lessac voice was dropped because its dataset is research-only. sherpa-onnx 2.x dropped espeak phonemization, so it is pinned to 1.12 |
| Interruption context | The heard text ends with a dash, explained by one system-prompt line | Tested on Qwen3-4B. A bracketed "(cut off…)" note made it restart the old answer or refuse; the dash worked in 6 of 6 trials |
| Streaming | Sentence-level chunks, with the LLM and TTS as concurrent tasks joined by a queue | First audio starts after the first sentence instead of after the whole reply |
| Interruption | Generation ids plus a server-side guard plus a client-side guard | Two independent guards mean stale audio can't play even if a cancel races |
| Memory | Assistant turns keep only sentences the client confirmed it started playing | Unheard text never enters context |

## Measured on this machine (Ryzen 5 5500U, 15 GB RAM, CPU only)

These are real measurements from development, not targets.

| Stage | Result |
|---|---|
| Kokoro int8: 2.8 s sentence | 6.2–6.8 s to synthesize, about 2.2× real time. Too slow, so rejected |
| Kokoro fp32, 4 threads: 2.8 s sentence | 2.15 s, about 0.77× real time |
| Kokoro fp32, 4 threads: "Sure." (0.64 s) | 0.66 s |
| Piper (VITS), 2 threads: 2.6 s sentence | 0.21 s, about 0.08× real time |
| Streaming STT + VAD compute | 0.107× real time |
| ASR finalize (flush) | 32–49 ms |
| Qwen3-4B Q4_K_M (llama-server, 6 threads): generation | 9–10 tokens/s |
| Qwen3-4B: prompt processing | about 40 tokens/s. A cold 210-token system prompt takes 5.3 s; with the prefix cached, 116 ms |

### End to end, with all real models (`tools/e2e_smoke.py`)

| Configuration | End of speech → final | Final → first token | **End of speech → first playback** | Gaps between sentences |
|---|---|---|---|---|
| Kokoro, before tuning | 1528 ms (hesitation-word bug) | 641 ms | 5.2 s | 3.9 s |
| Kokoro, after turn and chunk fixes | 760–782 ms | 647–781 ms | 3.0–4.6 s | **4.0 s and 8.9 s** (TTS slower than real time under LLM load) |
| **Piper (default), after fixes** | 737–748 ms | 856–929 ms | **2.0–3.2 s** | 0.2 s; or 2.1 s after a one-word opener |

Barge-in, measured end to end:
- **Stop:** playback stopped 12 ms after the partial transcript containing "No".
- **Memory:** kept only the sentence whose playback had started.
- **LLM:** llama-server logged `cancel task`.
- **New turn:** answered the correction.

Fixes made because of these measurements:
1. **Hesitation words:** prepositions removed. "…what is it famous for?" had waited 1.5 s instead of 0.7 s.
2. **Clause splits:** later chunks may now break at a clause after 40 characters, and dashes count even without spaces. This keeps synthesis ahead of playback.
3. **Abbreviations:** "p.m." or "etc." followed by a capitalized word now ends a sentence.
4. **Emphasis:** single-asterisk `*emphasis*` is stripped before TTS.
5. **Interruption marker:** the dash, as described above.

**Where the time goes now:**
- **LLM generation, about 9 tok/s.** It dominates "final → first sentence" and the gap after very short first chunks.
- **Turn-end wait:** the 700 ms silence threshold.

**Next levers:**
- A smaller model (Qwen2.5-1.5B, Apache-2.0, likely about 2× faster; not yet measured).
- GPU or Vulkan offload where available.
- Lowering `END_OF_TURN_SILENCE_MS` to 500–600, at the cost of more premature replies.

## Concurrency and cancellation

- **One asyncio loop.** CPU work (VAD, STT, TTS) runs in worker threads through `asyncio.to_thread`, so the WebSocket stays responsive.
- **Audio queue.** It is bounded at about 6 s and drops the oldest frames when full, so input stays real-time if the CPU falls behind. Drops are counted in metrics.
- **No reset races.** Pipeline resets are queued as markers, so they always run on the consumer task, never alongside a running `process()` call.
- **Interrupt order:**
  1. Clear `_active_gen`, so `on_sentence` and `on_audio` immediately drop anything the old task produces.
  2. Cancel the response task. The `ResponseStreamer` cancels both its LLM and TTS tasks, the HTTP stream closes, and llama-server stops.
  3. Send `interrupted`, and the browser stops all scheduled audio sources.
  4. Commit only heard sentences to memory, marked as cut off.
- **TTS already running** in a thread when the task is cancelled finishes in the background, and its result is discarded.

## Turn-taking state machine (`core/turn_detector.py`)

```
IDLE --speech--> SPEAKING --silence--> PAUSED --speech--> SPEAKING
                                         └── silence ≥ threshold ──▶ END_OF_TURN  (or DISCARD if voiced < MIN_UTTERANCE_MS)
```

**Barge-in while the assistant is active** (`core/barge_in.py`):
- An interruption needs **≥ 400 ms of voiced speech and ≥ 2 recognised words**, or one keyword ("wait", "stop", ...) after 200 ms.
- The words must not mostly match what the assistant is saying, which would mean echo.
- A finalized utterance that doesn't qualify (for example "yeah") is reported as `transcript_ignored` and is not answered.

## Extension points

- **Persistent memory:** implement `MemoryStore` (in `conversation_manager.py`) and consult it in `build_messages`.
- **Another STT engine:** implement `STTEngine`/`STTStream` (`speech/stt_base.py`).
  - A batch engine such as Whisper returns `""` from `partial()` and does its work in `finalize()`. That isn't streaming, so the UI would show no live partials.
- **Another TTS engine:** implement `TTSEngine` (`tts/tts_base.py`) and register it in `create_tts_engine`.
- **Another LLM:** any OpenAI-compatible server through env vars, or implement `LLMClient`.
- **Telephony:** implement `ClientTransport` (`core/transport.py`); see below.
- **Negotiation:** see `backend/app/negotiation/README.md`.

## Future: telephone calls (not implemented)

The engine (`VoiceSession`) only depends on `ClientTransport.send_event` and `send_audio`, plus `handle_audio(pcm16_16k)` and `handle_control`.

A `TelephonyTransport` would:
1. **Media in:** receive call audio from a SIP stack (e.g. a PJSIP or Asterisk ARI media bridge) or a provider media stream (typically 8 kHz μ-law over WebSocket). It decodes and upsamples to 16 kHz and calls `handle_audio`.
2. **Media out:** downsample TTS audio to 8 kHz μ-law and send it in about 20 ms packets, with its own jitter buffer.
3. **Playback acks:** emit `playback started/idle` itself, because the server now owns playback timing.
4. **Call control:** map call events (answer, DTMF, hang-up) to `start`, `stop` and `interrupt`.

Additional components needed before real calls:
- **Telephony audio:**
  - An 8 kHz narrowband ASR model, or upsampling plus an evaluation of the accuracy loss.
  - Echo cancellation on the server side; there is no browser AEC on a phone line.
- **Safety and compliance:**
  - An identity disclosure at call start ("This is an AI assistant calling on behalf of …").
  - A consent flow and jurisdiction-aware recording rules. Nothing is recorded by default today, and that must stay the default.
  - Authorization: who may place calls, to whom, and with what spending authority. See the negotiation design.
- **Operations:** rate limiting, an audit log of actions taken, and a human-takeover path.
