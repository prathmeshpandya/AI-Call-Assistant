# WebSocket protocol (`/ws`)

One WebSocket per conversation. Text frames are JSON; binary frames are audio.
In development the browser connects to `ws://localhost:5173/ws`, and Vite proxies it to the backend at `ws://127.0.0.1:8000/ws`.
Browsers must send an `Origin` listed in `FRONTEND_ORIGINS`. Other origins are rejected with close code 1008.

## Client → server

### Binary: microphone audio
- **Format:** PCM signed 16-bit little-endian, mono, **16 000 Hz**.
- **Frame size:** any even byte length up to `AUDIO_MAX_FRAME_BYTES` (default 32 000, which is 1 s). The browser sends 512 samples (1024 bytes, 32 ms) per frame.
- **When it's used:** audio is ignored unless the client has sent `start`.
- **Invalid frames** (empty, odd length, too large) get `error` with code `bad_audio`.

### JSON control messages
| `type` | Fields | Meaning |
|---|---|---|
| `start` | – | Microphone is on; begin listening |
| `stop` | – | Microphone is off; stops any reply and resets the input pipeline |
| `interrupt` | – | Manual interrupt (the same handling as a spoken barge-in) |
| `playback` | `event: "started"`, `generation_id`, `seq` | Playback of sentence `seq` began. Marks that sentence as **heard** |
| `playback` | `event: "idle"`, `generation_id` | All audio of the generation finished playing |
| `reset` | – | Stop any reply and clear conversation memory |
| `set_voice` | `voice` | Change the TTS voice, starting with the next reply |
| `ping` | `t` (number) | Liveness and RTT check; answered with `pong` |

A malformed message gets `error` with code `bad_message`; the connection stays open.

## Server → client

### Binary: assistant audio (one frame per sentence)
```
offset 0  uint32 LE  generation_id
offset 4  uint32 LE  seq            (0, 1, 2 ... within the generation)
offset 8  uint32 LE  sample_rate    (24000 for Kokoro, 22050 for the Piper voice)
offset 12 int16 LE[] mono PCM
```
The client must drop frames whose `generation_id` is not the current generation.

### JSON events
| `type` | Fields | Notes |
|---|---|---|
| `ready` | `session_id`, `input_sample_rate`, `tts_sample_rate`, `voices`, `voice`, `language`, `components{stt,tts,vad,llm}`, `config` | First message. Each component has `ok` and `detail` |
| `state` | `state`: `idle` \| `listening` \| `user_speaking` \| `thinking` \| `speaking` | Assistant status |
| `vad` | `event`: `speech_start` \| `discarded`, `reason?` | Discarded = noise, too short, or no words recognised |
| `transcript` | `final: false`, `text` | Provisional; replaces the previous partial. Never stored |
| `transcript` | `final: true`, `text`, `turn_id`, `user_turn_text` | Finalized user turn. `user_turn_text` is the merged user message when the user spoke twice before any reply was heard |
| `transcript_ignored` | `text`, `reason` | Speech during a reply that was not a real interruption (backchannel or echo) |
| `assistant_start` | `generation_id`, `turn_id` | New reply; the client resets its player to this generation |
| `assistant_text` | `generation_id`, `seq`, `text` | A sentence was sent to TTS; its audio follows |
| `assistant_end` | `generation_id`, `sentences` | Generation finished (all audio sent). The client replies `playback idle` when playback drains |
| `interrupted` | `generation_id`, `reason` (`barge_in` \| `manual` \| `mic_stopped` \| `reset`) | Stop playback **immediately** |
| `assistant_final` | `generation_id`, `interrupted`, `text` | What was committed to memory, which is only what was heard |
| `metrics` | `turn{...}`, `session{turns, interruptions, dropped_audio_frames}` | Latency for the completed turn (see below) |
| `voice_changed` | `voice` | |
| `history_reset` | – | |
| `pong` | `t` | Echo of the ping's `t` |
| `error` | `code`, `message`, `stage` | e.g. `llm_unavailable`/`llm`, `model_not_found`/`stt`, `bad_audio`/`audio` |

### Metrics fields (milliseconds, server clock `time.perf_counter`)
- **User speech:**
  - `speech_duration_ms`: from speech start to the last voiced audio.
  - `end_of_speech_to_final_ms`: from the last voiced audio to the final transcript. This includes the end-of-turn silence wait.
  - `asr_finalize_ms`: the recognizer flush alone.
- **Reply latency:** `final_to_first_token_ms`, `final_to_first_sentence_ms`, `final_to_first_audio_ms` (audio sent) and `final_to_first_playback_ms`. The last one is when the client's `playback started` notice arrived, so it includes browser scheduling and one local network hop.
- `end_of_speech_to_first_playback_ms`: the user-perceived response gap.
- **Audio totals:** `tts_ms[]` (time per sentence), `tts_total_ms`, `audio_total_ms` and `response_total_ms`.
- **Counters and status:** `sentences`, `tokens`, `interrupted` and `error`.

## Typical sequence
```
C: {"type":"start"}
C: <pcm> <pcm> ...                          (continuous)
S: {"type":"state","state":"listening"}
S: {"type":"vad","event":"speech_start"}
S: {"type":"transcript","final":false,"text":"What is the"}
S: {"type":"transcript","final":true,"text":"What is the capital of france", ...}
S: {"type":"assistant_start","generation_id":1}
S: {"type":"assistant_text","generation_id":1,"seq":0,"text":"The capital of France is Paris."}
S: <audio gen=1 seq=0>
C: {"type":"playback","event":"started","generation_id":1,"seq":0}
S: {"type":"state","state":"speaking"}
S: {"type":"assistant_end","generation_id":1,"sentences":2}
C: {"type":"playback","event":"idle","generation_id":1}
S: {"type":"assistant_final","generation_id":1,"interrupted":false,"text":"..."}
S: {"type":"metrics", ...}
```
