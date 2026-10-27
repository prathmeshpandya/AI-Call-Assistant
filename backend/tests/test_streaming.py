"""Response streaming, cancellation, LLM client errors and missing-model handling."""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest

from app.core.metrics import TurnMetrics
from app.core.response_streamer import ChunkerSettings, ResponseStreamer
from app.errors import LLMError, LLMUnavailableError, ModelNotFoundError
from app.llm.llm_base import ChatMessage
from app.llm.llm_service import OpenAICompatibleLLM
from app.services import Services
from app.speech.stt_service import create_stt_engine
from app.tts.tts_service import create_tts_engine
from tests.fakes import FakeLLM, FakeTTS, make_settings

MSGS = [ChatMessage("user", "hi")]


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, int, str]] = []

    async def on_sentence(self, generation_id, seq, text):
        self.events.append(("sentence", generation_id, seq, text))

    async def on_audio(self, generation_id, seq, audio):
        self.events.append(("audio", generation_id, seq, ""))


def streamer(llm, tts, sink):
    return ResponseStreamer(llm, tts, sink, ChunkerSettings(min_chars=5), voice="voice_a")


async def test_sentences_are_synthesized_in_order_without_overlap():
    llm = FakeLLM(["One sentence here. Second sentence here. Third and final one."], delay_s=0.005)
    tts = FakeTTS(delay_s=0.02)
    sink = RecordingSink()
    m = TurnMetrics(1)
    result = await streamer(llm, tts, sink).run(5, MSGS, m)
    assert result.sentences == 3 and result.error is None
    assert tts.calls == ["One sentence here.", "Second sentence here.", "Third and final one."]
    # Strict alternation sentence/audio per seq, ascending: ordered and never overlapping.
    assert [(e[0], e[2]) for e in sink.events] == [
        ("sentence", 0), ("audio", 0), ("sentence", 1), ("audio", 1), ("sentence", 2), ("audio", 2)]
    assert m.first_token <= m.first_sentence <= m.first_audio_sent <= m.response_done
    assert len(m.tts_ms) == 3


async def test_first_audio_arrives_before_generation_finishes():
    """Core streaming property: TTS of sentence 1 overlaps LLM generation of later text."""
    llm = FakeLLM(["Hi there. " + " ".join(["word"] * 60) + "."], delay_s=0.01)
    sink = RecordingSink()
    m = TurnMetrics(1)
    await streamer(llm, FakeTTS(), sink).run(1, MSGS, m)
    assert m.first_audio_sent is not None and m.response_done is not None
    # ~60 tokens * 10 ms remain to be generated after the first sentence
    assert (m.response_done - m.first_audio_sent) > 0.3


async def test_cancellation_stops_llm_and_tts():
    llm = FakeLLM(["First sentence is here. " + "more words " * 200], delay_s=0.01)
    tts = FakeTTS(delay_s=0.05)
    sink = RecordingSink()
    task = asyncio.create_task(streamer(llm, tts, sink).run(1, MSGS, TurnMetrics(1)))
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    n_events = len(sink.events)
    await asyncio.sleep(0.2)
    assert len(sink.events) == n_events  # nothing emitted after cancellation
    assert llm.cancelled == 1  # the LLM stream was closed


async def test_llm_failure_is_reported_not_faked():
    llm = FakeLLM(available=False)
    sink = RecordingSink()
    result = await streamer(llm, FakeTTS(), sink).run(1, MSGS, TurnMetrics(1))
    assert isinstance(result.error, LLMUnavailableError)
    assert result.sentences == 0 and sink.events == []


# --- OpenAI-compatible client -------------------------------------------------------------


def _sse(*chunks: str) -> bytes:
    lines = [f'data: {{"choices":[{{"delta":{{"content":"{c}"}}}}]}}\n\n' for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


async def test_llm_client_parses_sse_stream():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, content=_sse("Hello", " world", "."), headers={"content-type": "text/event-stream"})

    client = OpenAICompatibleLLM(make_settings(), transport=httpx.MockTransport(handler))
    assert [d async for d in client.stream_chat(MSGS)] == ["Hello", " world", "."]
    await client.aclose()


async def test_llm_client_http_error():
    client = OpenAICompatibleLLM(make_settings(), transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom")))
    with pytest.raises(LLMError, match="HTTP 500"):
        _ = [d async for d in client.stream_chat(MSGS)]
    await client.aclose()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def test_llm_server_unavailable_raises_clear_error():
    client = OpenAICompatibleLLM(make_settings(llm_base_url=f"http://127.0.0.1:{_free_port()}/v1"))
    with pytest.raises(LLMUnavailableError, match="Is llama-server running"):
        _ = [d async for d in client.stream_chat(MSGS)]
    health = await client.health()
    assert not health.ok and "not reachable" in health.detail
    await client.aclose()


# --- missing models -----------------------------------------------------------------------


def test_missing_models_raise_and_are_reported(tmp_path):
    s = make_settings(stt_model_dir=tmp_path / "nope-stt", tts_model_dir=tmp_path / "nope-tts",
                      vad_model_path=tmp_path / "nope.onnx")
    with pytest.raises(ModelNotFoundError):
        create_stt_engine(s)
    with pytest.raises(ModelNotFoundError):
        create_tts_engine(s)
    svc = Services.load(s)
    assert set(svc.errors) == {"stt", "tts", "vad"} and not svc.speech_ready
    assert "download-models.ps1" in svc.errors["stt"]
