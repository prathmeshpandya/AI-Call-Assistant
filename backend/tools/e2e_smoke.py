"""End-to-end check against a RUNNING backend with the real models.

Speaks a question into /ws exactly like the browser does (16 kHz PCM16, 32 ms frames, real-time pace),
then prints every event, the latency metrics, and saves the assistant's audio to a WAV file.

The input speech is generated with the local TTS model (or read from --wav), so this exercises real
VAD + streaming STT + LLM + TTS, but not the browser's microphone capture or speaker playback.
Playback notices are sent when each audio frame is *received*, so "first_playback" here means
"first audio received by the client".

Usage (backend running, from backend/):
    .venv\\Scripts\\python tools\\e2e_smoke.py --text "What is the capital of France?"
    .venv\\Scripts\\python tools\\e2e_smoke.py --wav path\\to\\question.wav --barge-in "No wait, tell me about Germany instead."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.audio_utils import float32_to_pcm16, pcm16_to_float32, resample_linear, unpack_audio_frame  # noqa: E402
from app.config import get_settings  # noqa: E402

FRAME = 512
RATE = 16000


def synth_question(text: str) -> np.ndarray:
    from app.tts.tts_service import create_tts_engine

    s = get_settings()
    tts = create_tts_engine(s)
    voice = s.tts_voice if s.tts_voice in tts.voices() else tts.voices()[0]
    audio = tts.synthesize(text, voice)
    return resample_linear(audio.samples, audio.sample_rate, RATE)


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise SystemExit("WAV must be 16-bit PCM")
        data = pcm16_to_float32(w.readframes(w.getnframes()))
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        return resample_linear(data, w.getframerate(), RATE)


async def send_speech(ws, samples: np.ndarray, trailing_silence_s: float = 2.0) -> None:
    samples = np.concatenate([np.zeros(int(0.3 * RATE), np.float32), samples,
                              np.zeros(int(trailing_silence_s * RATE), np.float32)])
    start = time.perf_counter()
    for i in range(0, len(samples) - FRAME + 1, FRAME):
        await ws.send(float32_to_pcm16(samples[i : i + FRAME]))
        # pace at real time like a microphone
        delay = start + (i + FRAME) / RATE - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)


async def notify_later(ws, delay_s: float, message: dict) -> None:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    await ws.send(json.dumps(message))


async def run(args) -> int:
    question = load_wav(Path(args.wav)) if args.wav else synth_question(args.text)
    barge = synth_question(args.barge_in) if args.barge_in else None
    out_audio: list[np.ndarray] = []
    out_rate = 24000
    # Simulated speaker: each chunk plays at max(arrival, end of previous chunk). A positive
    # difference is an audible gap (playback underrun) between sentences.
    play_end: float | None = None
    gaps: list[float] = []
    notices: list[asyncio.Task] = []
    report: dict = {"transcripts": [], "replies": [], "metrics": [], "gaps_ms": []}
    t0 = time.perf_counter()
    stamp = lambda: f"{(time.perf_counter() - t0) * 1000:8.0f} ms"  # noqa: E731

    async with websockets.connect(args.url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print(stamp(), "READY", json.dumps(ready["components"], indent=None))
        await ws.send(json.dumps({"type": "start"}))
        sender = asyncio.create_task(send_speech(ws, question))
        finals = 0
        barge_sent = False
        deadline = time.perf_counter() + args.timeout
        while time.perf_counter() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=deadline - time.perf_counter())
            except TimeoutError:
                break
            if isinstance(msg, bytes):
                gen, seq, rate, pcm = unpack_audio_frame(msg)
                out_rate = rate
                out_audio.append(pcm16_to_float32(pcm))
                dur = len(pcm) / 2 / rate
                arrived = time.perf_counter()
                if seq == 0 or play_end is None:
                    start_at = arrived
                    gap = 0.0
                else:
                    gap = max(0.0, arrived - play_end)
                    gaps.append(gap)
                    start_at = max(arrived, play_end)
                play_end = start_at + dur
                print(stamp(), f"AUDIO gen={gen} seq={seq} {dur:.2f}s" + (f"  (gap before it: {gap * 1000:.0f} ms)" if seq else ""))
                # Report "started" when the simulated speaker would actually start it, like the browser does.
                notices.append(asyncio.create_task(
                    notify_later(ws, start_at - arrived, {"type": "playback", "event": "started", "generation_id": gen, "seq": seq})))
                if barge is not None and not barge_sent:
                    barge_sent = True
                    print(stamp(), "-- speaking over the assistant (barge-in test)")
                    sender = asyncio.create_task(send_speech(ws, barge))
                continue
            ev = json.loads(msg)
            kind = ev["type"]
            if kind == "transcript" and not ev["final"]:
                print(stamp(), "partial:", ev["text"])
            elif kind == "metrics":
                print(stamp(), "METRICS", json.dumps(ev["turn"], indent=2))
                report["metrics"].append(ev["turn"])
            else:
                print(stamp(), kind.upper(), {k: v for k, v in ev.items() if k != "type"})
            if kind == "assistant_end":
                remaining = (play_end or 0) - time.perf_counter()
                notices.append(asyncio.create_task(
                    notify_later(ws, remaining, {"type": "playback", "event": "idle", "generation_id": ev["generation_id"]})))
            if kind == "interrupted":
                for task in notices:  # the speaker stops: nothing further "starts"
                    task.cancel()
                notices.clear()
                play_end = None
            if kind == "transcript" and ev["final"]:
                report["transcripts"].append(ev["text"])
            if kind == "assistant_final":
                report["replies"].append({"text": ev["text"], "interrupted": ev["interrupted"]})
                finals += 1
                if finals >= (2 if barge is not None else 1):
                    # wait briefly for the metrics event that follows
                    try:
                        m = json.loads(await asyncio.wait_for(ws.recv(), 2))
                        if m.get("type") == "metrics":
                            print(stamp(), "METRICS", json.dumps(m["turn"], indent=2))
                            report["metrics"].append(m["turn"])
                    except (TimeoutError, json.JSONDecodeError, TypeError):
                        pass
                    break
        sender.cancel()

    report["gaps_ms"] = [round(g * 1000) for g in gaps]
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if gaps:
        print(f"Simulated playback gaps between sentences: {', '.join(f'{g * 1000:.0f}' for g in gaps)} ms "
              f"(total {sum(gaps) * 1000:.0f} ms)")
    if out_audio:
        path = Path(args.save)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(out_rate)
            w.writeframes(float32_to_pcm16(np.concatenate(out_audio)))
        print(f"Saved assistant audio to {path.resolve()}")
        return 0
    print("No assistant audio received.")
    return 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--text", default="What is the capital of France?")
    p.add_argument("--wav", help="16-bit PCM WAV with the user's question (instead of --text)")
    p.add_argument("--barge-in", help="second utterance spoken over the first assistant reply")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--json-out", help="write transcripts, replies, metrics and gaps to this JSON file")
    p.add_argument("--save", default="e2e_assistant_reply.wav")
    raise SystemExit(asyncio.run(run(p.parse_args())))


if __name__ == "__main__":
    main()
