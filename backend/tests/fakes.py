"""Deterministic stand-ins for the model engines, implementing the same interfaces.

Used only in unit tests; the real engines are exercised by tools/e2e_smoke.py.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np

from app.audio.vad import VoiceActivityDetector
from app.config import Settings
from app.errors import LLMUnavailableError
from app.llm.llm_base import ChatMessage, LLMClient, LLMHealth
from app.services import Services
from app.speech.stt_base import STTEngine, STTStream
from app.tts.tts_base import SynthesizedAudio, TTSEngine

FRAME = 512  # samples per VAD window at 16 kHz


def loud(n_frames: int) -> bytes:
    """PCM16 'speech' (a 220 Hz tone at -6 dBFS)."""
    t = np.arange(n_frames * FRAME) / 16000
    return (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


def silence(n_frames: int) -> bytes:
    return np.zeros(n_frames * FRAME, dtype="<i2").tobytes()


class FakeVAD(VoiceActivityDetector):
    window_size = FRAME

    def accept(self, window: np.ndarray) -> bool:
        return float(np.sqrt(np.mean(window**2))) > 0.05

    def reset(self) -> None:
        pass


class FakeSTTStream(STTStream):
    """Reveals one word of the scripted transcript per 4 voiced windows."""

    def __init__(self, engine: "FakeSTT") -> None:
        self._engine = engine
        self._voiced = 0

    def accept(self, samples: np.ndarray) -> None:
        if float(np.sqrt(np.mean(samples**2))) > 0.05:
            self._voiced += 1

    def _words(self) -> list[str]:
        return self._engine.next_transcript().split()

    def partial(self) -> str:
        return " ".join(self._words()[: self._voiced // 4])

    def finalize(self) -> str:
        text = " ".join(self._words()[: max(1, self._voiced // 4)]) if self._voiced else ""
        self._engine.consume()
        return text

    def reset(self) -> None:
        self._voiced = 0


class FakeSTT(STTEngine):
    name = "fake-stt"
    sample_rate = 16000
    is_streaming = True

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)

    def next_transcript(self) -> str:
        return self._transcripts[0] if self._transcripts else ""

    def consume(self) -> None:
        if self._transcripts:
            self._transcripts.pop(0)

    def create_stream(self) -> STTStream:
        return FakeSTTStream(self)


class FakeTTS(TTSEngine):
    name = "fake-tts"
    sample_rate = 24000

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls: list[str] = []

    def voices(self) -> list[str]:
        return ["voice_a", "voice_b"]

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> SynthesizedAudio:
        self.calls.append(text)
        if self.delay_s:
            time.sleep(self.delay_s)
        return SynthesizedAudio(np.full(240 * len(text), 0.1, dtype=np.float32), self.sample_rate)


class FakeLLM(LLMClient):
    def __init__(self, replies: list[str] | None = None, delay_s: float = 0.0, available: bool = True) -> None:
        self.replies = list(replies or ["Hello there. How can I help you today?"])
        self.delay_s = delay_s
        self.available = available
        self.requests: list[list[ChatMessage]] = []
        self.cancelled = 0

    async def health(self) -> LLMHealth:
        return LLMHealth(ok=self.available, model="fake", detail="" if self.available else "down")

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.requests.append(messages)
        if not self.available:
            raise LLMUnavailableError("Cannot reach the LLM server")
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        try:
            for word in reply.split(" "):
                if self.delay_s:
                    await asyncio.sleep(self.delay_s)
                yield word + " "
        except (asyncio.CancelledError, GeneratorExit):
            self.cancelled += 1
            raise


def make_settings(**overrides) -> Settings:
    base = dict(
        warmup=False,
        end_of_turn_silence_ms=300,
        end_of_turn_hesitation_silence_ms=600,
        min_utterance_ms=100,
        vad_min_speech_ms=0,
        barge_in_min_speech_ms=200,
        barge_in_min_words=2,
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


class FakeServices(Services):
    def create_vad(self) -> VoiceActivityDetector:
        return FakeVAD()


def make_services(settings: Settings | None = None, *, transcripts=None, llm=None, tts=None) -> FakeServices:
    return FakeServices(
        settings=settings or make_settings(),
        llm=llm or FakeLLM(),
        system_prompt="You are a test assistant.",
        stt=FakeSTT(transcripts or ["what is the weather like today"]),
        tts=tts or FakeTTS(),
    )
