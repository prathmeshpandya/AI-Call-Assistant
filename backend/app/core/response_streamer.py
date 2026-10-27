"""Sentence-level response streaming: LLM tokens -> sentence chunks -> TTS -> audio sink.

    [LLM producer task] --tokens--> SentenceChunker --sentences--> asyncio.Queue --> [TTS consumer task] --> sink

The producer and consumer run concurrently, so the LLM keeps generating while earlier sentences
are synthesized and played. The queue preserves order and the single consumer guarantees that
audio is emitted strictly in sentence order with no overlap.

Cancellation: cancelling `run()` cancels both tasks. The LLM HTTP stream is closed (llama-server
stops generating) and any in-flight TTS result is discarded because the awaiting coroutine is gone.
The sink additionally checks the generation id so stale audio can never reach the client.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from app.audio.audio_utils import duration_ms
from app.core.metrics import TurnMetrics, now
from app.core.sentence_chunker import SentenceChunker, clean_for_speech
from app.errors import VoiceAssistantError
from app.llm.llm_base import ChatMessage, LLMClient
from app.tts.tts_base import SynthesizedAudio, TTSEngine

log = logging.getLogger(__name__)

_END = object()


class ResponseSink(Protocol):
    async def on_sentence(self, generation_id: int, seq: int, text: str) -> None: ...
    async def on_audio(self, generation_id: int, seq: int, audio: SynthesizedAudio) -> None: ...


@dataclass
class ChunkerSettings:
    min_chars: int = 12
    max_chars: int = 220
    first_soft_chars: int = 30
    soft_chars: int = 40


@dataclass
class ResponseResult:
    generated_text: str
    sentences: int
    error: VoiceAssistantError | None = None


class ResponseStreamer:
    def __init__(
        self,
        llm: LLMClient,
        tts: TTSEngine,
        sink: ResponseSink,
        chunking: ChunkerSettings,
        voice: str,
        speed: float = 1.0,
    ) -> None:
        self._llm = llm
        self._tts = tts
        self._sink = sink
        self._chunking = chunking
        self._voice = voice
        self._speed = speed

    async def run(self, generation_id: int, messages: list[ChatMessage], metrics: TurnMetrics) -> ResponseResult:
        queue: asyncio.Queue[object] = asyncio.Queue()
        state = {"text": "", "error": None}
        producer = asyncio.create_task(self._produce(messages, queue, metrics, state), name=f"llm-{generation_id}")
        consumer = asyncio.create_task(self._consume(generation_id, queue, metrics), name=f"tts-{generation_id}")
        try:
            await producer
            seq_count = await consumer
        except BaseException:  # includes CancelledError from an interruption
            producer.cancel()
            consumer.cancel()
            await asyncio.gather(producer, consumer, return_exceptions=True)
            raise
        metrics.response_done = now()
        return ResponseResult(state["text"], seq_count, state["error"])  # type: ignore[arg-type]

    async def _produce(self, messages, queue, metrics: TurnMetrics, state: dict) -> None:
        chunker = SentenceChunker(**self._chunking.__dict__)
        metrics.llm_request = now()
        try:
            async for delta in self._llm.stream_chat(messages):
                if metrics.first_token is None:
                    metrics.first_token = now()
                metrics.tokens += 1  # one SSE delta is ~one token for llama-server
                state["text"] += delta
                for sentence in chunker.feed(delta):
                    await self._enqueue(sentence, queue, metrics)
            tail = chunker.flush()
            if tail:
                await self._enqueue(tail, queue, metrics)
        except VoiceAssistantError as exc:
            # Let already-queued sentences finish; the session reports the error.
            log.warning("LLM generation failed", extra={"error": str(exc)})
            state["error"] = exc
        finally:
            await queue.put(_END)

    async def _enqueue(self, sentence: str, queue: asyncio.Queue, metrics: TurnMetrics) -> None:
        speakable = clean_for_speech(sentence)
        if not speakable:
            return
        if metrics.first_sentence is None:
            metrics.first_sentence = now()
        await queue.put(speakable)

    async def _consume(self, generation_id: int, queue: asyncio.Queue, metrics: TurnMetrics) -> int:
        seq = 0
        while True:
            item = await queue.get()
            if item is _END:
                return seq
            sentence = str(item)
            await self._sink.on_sentence(generation_id, seq, sentence)
            t0 = now()
            audio = await asyncio.to_thread(self._tts.synthesize, sentence, self._voice, self._speed)
            metrics.tts_ms.append((now() - t0) * 1000.0)
            metrics.audio_ms.append(duration_ms(len(audio.samples), audio.sample_rate))
            await self._sink.on_audio(generation_id, seq, audio)
            if metrics.first_audio_sent is None:
                metrics.first_audio_sent = now()
            metrics.sentences += 1
            seq += 1
