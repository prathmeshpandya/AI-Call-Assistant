"""WebSocket protocol and full-session behaviour with fake engines (real VAD/STT/TTS/LLM are not used)."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.audio.audio_utils import unpack_audio_frame
from app.main import create_app
from tests.fakes import FakeLLM, FakeTTS, loud, make_services, make_settings, silence


class Client:
    """Small helper around the Starlette test WebSocket that splits JSON events and audio frames."""

    def __init__(self, ws):
        self.ws = ws
        self.events: list[dict] = []
        self.audio: list[tuple[int, int]] = []

    def send(self, **msg):
        self.ws.send_text(json.dumps(msg))

    def recv(self):
        msg = self.ws.receive()
        if msg.get("bytes") is not None:
            gen, seq, rate, pcm = unpack_audio_frame(msg["bytes"])
            self.audio.append((gen, seq))
            return ("audio", gen, seq)
        event = json.loads(msg["text"])
        self.events.append(event)
        return event

    def until(self, pred, limit=500):
        for _ in range(limit):
            item = self.recv()
            if isinstance(item, dict) and pred(item):
                return item
        raise AssertionError(f"condition not met; events={[e.get('type') for e in self.events]}")

    def speak(self, voiced_frames=24, silent_frames=15):
        self.ws.send_bytes(loud(voiced_frames))
        self.ws.send_bytes(silence(silent_frames))


def make_client(**kw):
    settings = kw.pop("settings", None) or make_settings()
    app = create_app(settings, make_services(settings, **kw))
    return TestClient(app)


def test_connect_ready_and_disconnect_cleanup():
    with make_client() as tc:
        with tc.websocket_connect("/ws") as ws:
            ready = json.loads(ws.receive_text())
            assert ready["type"] == "ready"
            assert ready["voices"] == ["voice_a", "voice_b"] and ready["tts_sample_rate"] == 24000
            assert ready["components"]["llm"]["ok"]
            assert tc.app.state.sessions.count == 1
        for _ in range(50):  # session cleanup happens in the server task after disconnect
            if tc.app.state.sessions.count == 0:
                break
            time.sleep(0.02)
        assert tc.app.state.sessions.count == 0


def test_rejects_foreign_origin():
    with make_client() as tc:
        with pytest.raises(WebSocketDisconnect):
            with tc.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
                ws.receive_text()


def test_invalid_messages_get_errors_not_crashes():
    with make_client() as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        c.until(lambda e: e["type"] == "ready")
        c.send(type="bogus")
        assert c.until(lambda e: e["type"] == "error")["code"] == "bad_message"
        c.send(type="start")
        ws.send_bytes(b"\x01")  # odd byte count
        assert c.until(lambda e: e["type"] == "error")["code"] == "bad_audio"
        c.send(type="ping", t=1)
        assert c.until(lambda e: e["type"] == "pong")["t"] == 1


def test_full_voice_turn_streams_text_and_audio():
    llm = FakeLLM(["Sure thing. The weather is sunny and warm today."])
    with make_client(llm=llm) as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        c.until(lambda e: e["type"] == "ready")
        c.send(type="start")
        c.speak()
        final = c.until(lambda e: e["type"] == "transcript" and e["final"])
        assert final["text"] == "what is the weather like today"
        gen = c.until(lambda e: e["type"] == "assistant_start")["generation_id"]
        end = c.until(lambda e: e["type"] == "assistant_end")
        assert end["sentences"] == 2 and c.audio == [(gen, 0), (gen, 1)]
        texts = [e["text"] for e in c.events if e["type"] == "assistant_text"]
        assert texts == ["Sure thing.", "The weather is sunny and warm today."]
        # Client reports playback; the turn completes and is committed to memory.
        c.send(type="playback", event="started", generation_id=gen, seq=0)
        c.send(type="playback", event="started", generation_id=gen, seq=1)
        c.send(type="playback", event="idle", generation_id=gen)
        done = c.until(lambda e: e["type"] == "assistant_final")
        assert done == {"type": "assistant_final", "generation_id": gen, "interrupted": False,
                        "text": "Sure thing. The weather is sunny and warm today."}
        metrics = c.until(lambda e: e["type"] == "metrics")["turn"]
        assert metrics["final_to_first_token_ms"] is not None and metrics["final_to_first_playback_ms"] is not None
        assert llm.requests[0][-1].content == "what is the weather like today"


def test_partial_transcripts_never_enter_history():
    llm = FakeLLM(["Okay. Noted."])
    with make_client(llm=llm) as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        c.until(lambda e: e["type"] == "ready")
        c.send(type="start")
        c.speak()
        c.until(lambda e: e["type"] == "assistant_end")
        partials = [e for e in c.events if e["type"] == "transcript" and not e["final"]]
        assert len(partials) >= 2  # incremental updates were shown...
        user_msgs = [m for m in llm.requests[0] if m.role == "user"]
        assert [m.content for m in user_msgs] == ["what is the weather like today"]  # ...but stored once


def test_barge_in_interrupts_and_blocks_stale_audio():
    # Slow LLM + TTS so the first response is still in flight when the user interrupts.
    llm = FakeLLM(["I think the best approach would be to start by planning. " + "Then more words. " * 30,
                   "Sure, the other approach is to iterate quickly."], delay_s=0.02)
    tts = FakeTTS(delay_s=0.05)
    transcripts = ["tell me the best approach", "no wait explain the other approach"]
    with make_client(llm=llm, tts=tts, transcripts=transcripts) as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        c.until(lambda e: e["type"] == "ready")
        c.send(type="start")
        c.speak()
        gen1 = c.until(lambda e: e["type"] == "assistant_start")["generation_id"]
        c.until(lambda e: e["type"] == "assistant_text")
        c.send(type="playback", event="started", generation_id=gen1, seq=0)
        c.speak(voiced_frames=24, silent_frames=15)  # user talks over the assistant
        intr = c.until(lambda e: e["type"] == "interrupted")
        assert intr == {"type": "interrupted", "generation_id": gen1, "reason": "barge_in"}
        final1 = c.until(lambda e: e["type"] == "assistant_final" and e["generation_id"] == gen1)
        assert final1["interrupted"] and final1["text"] == "I think the best approach would be to start by planning."
        n_audio_gen1 = sum(1 for g, _ in c.audio if g == gen1)
        gen2 = c.until(lambda e: e["type"] == "assistant_start")["generation_id"]
        assert gen2 > gen1
        c.until(lambda e: e["type"] == "assistant_end" and e["generation_id"] == gen2)
        # No audio for the interrupted generation arrived after the interruption.
        assert sum(1 for g, _ in c.audio if g == gen1) == n_audio_gen1
        assert any(g == gen2 for g, _ in c.audio)
        # Context for the new reply contains only what was heard, marked as cut off.
        second = llm.requests[1]
        assert [m.role for m in second] == ["system", "user", "assistant", "user"]
        assert "Then more words" not in second[2].content and second[2].content.endswith("—")
        assert second[3].content == "no wait explain the other approach"
        assert llm.cancelled >= 1


def test_manual_interrupt_and_repeated_interruptions_stay_responsive():
    llm = FakeLLM(["Long answer sentence one. " + "Filler words here. " * 40], delay_s=0.01)
    transcripts = [f"question number {i}" for i in range(3)]
    with make_client(llm=llm, transcripts=transcripts) as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        c.until(lambda e: e["type"] == "ready")
        c.send(type="start")
        for _ in range(3):
            c.speak()
            gen = c.until(lambda e: e["type"] == "assistant_start")["generation_id"]
            c.until(lambda e: e["type"] == "assistant_text")
            c.send(type="interrupt")
            assert c.until(lambda e: e["type"] == "interrupted")["generation_id"] == gen
            c.until(lambda e: e["type"] == "assistant_final")
        c.send(type="ping", t=2)
        assert c.until(lambda e: e["type"] == "pong")["t"] == 2
        metrics = [e for e in c.events if e["type"] == "metrics"]
        assert metrics[-1]["session"]["interruptions"] == 3


def test_backchannel_during_speech_is_not_a_turn():
    llm = FakeLLM(["First sentence of a long answer. " + "More text. " * 40], delay_s=0.02)
    with make_client(llm=llm, transcripts=["tell me a story", "yeah"]) as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        c.until(lambda e: e["type"] == "ready")
        c.send(type="start")
        c.speak()
        c.until(lambda e: e["type"] == "assistant_text")
        c.speak(voiced_frames=6, silent_frames=15)  # short "yeah"
        ignored = c.until(lambda e: e["type"] in ("transcript_ignored", "interrupted"))
        assert ignored["type"] == "transcript_ignored"
        # The reply is still streaming; stop it so TestClient teardown doesn't race the cancellation.
        c.send(type="stop")
        c.until(lambda e: e["type"] == "interrupted")


def test_noise_and_empty_transcripts_are_discarded():
    with make_client(transcripts=[""]) as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        c.until(lambda e: e["type"] == "ready")
        c.send(type="start")
        c.speak(voiced_frames=2, silent_frames=15)  # 64 ms blip
        assert c.until(lambda e: e["type"] == "vad" and e["event"] == "discarded")["reason"] == "too short"
        c.speak(voiced_frames=10, silent_frames=15)  # long enough, but no words recognised
        assert c.until(lambda e: e["type"] == "vad" and e["event"] == "discarded")["reason"] == "no words recognised"
        assert not any(e["type"] == "assistant_start" for e in c.events)


def test_llm_unavailable_surfaces_error_and_session_recovers():
    llm = FakeLLM(available=False)
    with make_client(llm=llm) as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        ready = c.until(lambda e: e["type"] == "ready")
        assert not ready["components"]["llm"]["ok"]
        c.send(type="start")
        c.speak()
        err = c.until(lambda e: e["type"] == "error")
        assert err["code"] == "llm_unavailable" and err["stage"] == "llm"
        assert c.until(lambda e: e["type"] == "assistant_final")["text"] == ""
        assert c.until(lambda e: e["type"] == "state")["state"] == "listening"


def test_reset_clears_history_and_voice_change():
    llm = FakeLLM(["Okay."])
    with make_client(llm=llm, transcripts=["first question", "second question"]) as tc, tc.websocket_connect("/ws") as ws:
        c = Client(ws)
        c.until(lambda e: e["type"] == "ready")
        c.send(type="set_voice", voice="voice_b")
        assert c.until(lambda e: e["type"] == "voice_changed")["voice"] == "voice_b"
        c.send(type="set_voice", voice="nope")
        assert c.until(lambda e: e["type"] == "error")["code"] == "bad_voice"
        c.send(type="start")
        c.speak()
        gen = c.until(lambda e: e["type"] == "assistant_end")["generation_id"]
        c.send(type="playback", event="idle", generation_id=gen)
        c.until(lambda e: e["type"] == "assistant_final")
        c.send(type="reset")
        c.until(lambda e: e["type"] == "history_reset")
        c.speak()
        c.until(lambda e: e["type"] == "assistant_start")
        assert [m.role for m in llm.requests[-1]] == ["system", "user"]
