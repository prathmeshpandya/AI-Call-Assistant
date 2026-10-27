"""One live voice conversation: owns the audio pipeline, the in-flight response and the memory.

Concurrency model (all on one asyncio loop, CPU work in worker threads):
* audio consumer task - drains the inbound PCM queue, runs VAD + STT, reacts to turn events;
* response task       - one ResponseStreamer run (LLM + TTS) per assistant reply.

Every reply gets a monotonically increasing generation id. `_active_gen` is the only generation
allowed to send text/audio; an interruption clears it *before* cancelling the task, so anything the
old task still produces is dropped (stale-audio guard). The client tags playback notices with the
generation id, and notices for old generations are ignored.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from enum import Enum
from typing import Any

from app.audio.audio_utils import AudioFormatError, pcm16_to_float32, validate_pcm16_frame
from app.config import Settings
from app.core.audio_pipeline import (
    AudioPipeline,
    FinalTranscript,
    PartialTranscript,
    PipelineError,
    PipelineEvent,
    SpeechStarted,
    UtteranceDiscarded,
)
from app.core.barge_in import BargeInConfig, should_barge_in
from app.core.conversation_manager import ConversationManager
from app.core.metrics import TurnMetrics, now
from app.core.response_streamer import ChunkerSettings, ResponseResult, ResponseStreamer
from app.core.transport import ClientTransport
from app.core.turn_detector import TurnConfig
from app.errors import VoiceAssistantError
from app.services import Services
from app.tts.tts_base import SynthesizedAudio

log = logging.getLogger(__name__)

_RESET = object()  # queued marker: reset the pipeline from the consumer task (no thread races)
_AUDIO_QUEUE_FRAMES = 200  # ~6 s of 32 ms frames; overflow means the CPU cannot keep up


class AssistantState(str, Enum):
    IDLE = "idle"  # microphone off
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceSession:
    def __init__(self, session_id: str, services: Services, transport: ClientTransport) -> None:
        self.id = session_id
        self._services = services
        self._s: Settings = services.settings
        self._transport = transport
        self._conv = ConversationManager(self._s.max_history_turns, self._s.max_history_chars)
        self._audio_q: asyncio.Queue[object] = asyncio.Queue(maxsize=_AUDIO_QUEUE_FRAMES)
        self._pipeline: AudioPipeline | None = None
        self._listening = False
        self._state = AssistantState.IDLE
        self._generation = 0
        self._active_gen: int | None = None
        self._response_task: asyncio.Task | None = None
        self._response_done: set[int] = set()
        self._consumer_task: asyncio.Task | None = None
        self._metrics: dict[int, TurnMetrics] = {}
        self._turn_counter = 0
        self._interruptions = 0
        self._dropped_frames = 0
        self._voice = services.default_voice()
        self._barge_cfg = BargeInConfig(
            enabled=self._s.barge_in_enabled,
            min_speech_ms=self._s.barge_in_min_speech_ms,
            min_words=self._s.barge_in_min_words,
            keywords=self._s.barge_in_keyword_list,
        )
        self._closed = False

    # --- lifecycle ----------------------------------------------------------------------------

    async def start(self) -> None:
        if self._services.speech_ready:
            assert self._services.stt is not None
            self._pipeline = AudioPipeline(
                vad=self._services.create_vad(),
                stt=self._services.stt.create_stream(),
                turn_config=TurnConfig(
                    end_of_turn_silence_ms=self._s.end_of_turn_silence_ms,
                    hesitation_silence_ms=self._s.end_of_turn_hesitation_silence_ms,
                    min_utterance_ms=self._s.min_utterance_ms,
                    max_utterance_ms=int(self._s.max_utterance_s * 1000),
                ),
                sample_rate=self._s.audio_input_sample_rate,
                preroll_ms=self._s.preroll_ms,
                vad_trigger_delay_ms=self._s.vad_min_speech_ms,
            )
            self._consumer_task = asyncio.create_task(self._consume_audio(), name=f"audio-{self.id}")
        ready = await self._ready_event()
        await self._send(ready)
        for component, detail in self._services.errors.items():
            await self._error("model_not_found", detail, stage=component)
        llm = ready["components"]["llm"]
        if not llm["ok"]:
            await self._error("llm_unavailable", llm["detail"], stage="llm")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._active_gen = None
        for task in (self._response_task, self._consumer_task):
            if task and not task.done():
                task.cancel()
        await asyncio.gather(
            *(t for t in (self._response_task, self._consumer_task) if t), return_exceptions=True
        )
        log.info("Session closed", extra={"session": self.id, "turns": self._turn_counter,
                                          "interruptions": self._interruptions})

    async def _ready_event(self) -> dict[str, Any]:
        health = await self._services.llm.health()
        tts = self._services.tts
        comp = lambda name: {"ok": name not in self._services.errors, "detail": self._services.errors.get(name, "")}  # noqa: E731
        return {
            "type": "ready",
            "session_id": self.id,
            "input_sample_rate": self._s.audio_input_sample_rate,
            "tts_sample_rate": tts.sample_rate if tts else None,
            "voices": tts.voices() if tts else [],
            "voice": self._voice,
            "language": self._s.language,
            "components": {
                "stt": {**comp("stt"), "engine": self._services.stt.name if self._services.stt else None},
                "tts": {**comp("tts"), "engine": tts.name if tts else None},
                "vad": comp("vad"),
                "llm": {"ok": health.ok, "detail": health.detail, "model": health.model},
            },
            "config": {
                "end_of_turn_silence_ms": self._s.end_of_turn_silence_ms,
                "barge_in_enabled": self._s.barge_in_enabled,
            },
        }

    # --- inbound audio ------------------------------------------------------------------------

    async def handle_audio(self, data: bytes) -> None:
        if not self._listening or self._pipeline is None:
            return
        try:
            validate_pcm16_frame(data, self._s.audio_max_frame_bytes)
        except AudioFormatError as exc:
            await self._error("bad_audio", str(exc), stage="audio")
            return
        try:
            self._audio_q.put_nowait(data)
        except asyncio.QueueFull:
            self._dropped_frames += 1
            with contextlib.suppress(asyncio.QueueEmpty):
                self._audio_q.get_nowait()  # drop the oldest frame to stay real-time
            self._audio_q.put_nowait(data)
            if self._dropped_frames % 50 == 1:
                log.warning("Audio backlog; dropping frames", extra={"session": self.id, "dropped": self._dropped_frames})

    async def _consume_audio(self) -> None:
        assert self._pipeline is not None
        while True:
            item = await self._audio_q.get()
            if item is _RESET:
                self._pipeline.reset()
                continue
            samples = pcm16_to_float32(item)  # type: ignore[arg-type]
            try:
                events = await asyncio.to_thread(self._pipeline.process, samples)
            except Exception:
                log.exception("Audio pipeline failure", extra={"session": self.id})
                await self._error("audio_pipeline_failed", "Audio processing failed", stage="audio")
                continue
            for event in events:
                try:
                    await self._on_pipeline_event(event)
                except Exception:
                    log.exception("Event handling failed", extra={"session": self.id})

    async def _on_pipeline_event(self, ev: PipelineEvent) -> None:
        assistant_active = self._active_gen is not None
        if isinstance(ev, SpeechStarted):
            await self._send({"type": "vad", "event": "speech_start"})
            if not assistant_active:
                await self._set_state(AssistantState.USER_SPEAKING)
        elif isinstance(ev, PartialTranscript):
            await self._send({"type": "transcript", "final": False, "text": ev.text})
            if assistant_active and should_barge_in(
                ev.text, ev.speech_ms, self._conv.current_assistant_text(), self._barge_cfg
            ):
                await self.interrupt("barge_in")
                await self._set_state(AssistantState.USER_SPEAKING)
        elif isinstance(ev, FinalTranscript):
            if assistant_active:
                if should_barge_in(ev.text, ev.speech_ms, self._conv.current_assistant_text(), self._barge_cfg):
                    await self.interrupt("barge_in")
                else:
                    # Short backchannel ("yeah"), noise, or the assistant's own echo: not a new turn.
                    await self._send({"type": "transcript_ignored", "text": ev.text, "reason": "not_an_interruption"})
                    return
            await self._handle_user_turn(ev)
        elif isinstance(ev, UtteranceDiscarded):
            await self._send({"type": "vad", "event": "discarded", "reason": ev.reason})
            if not assistant_active:
                await self._set_state(self._idle_state())
        elif isinstance(ev, PipelineError):
            await self._error("stt_failed", ev.message, stage="stt")
            if not assistant_active:
                await self._set_state(self._idle_state())

    # --- turns and responses ------------------------------------------------------------------

    async def _handle_user_turn(self, ev: FinalTranscript) -> None:
        try:
            turn = self._conv.add_user(ev.text)
        except ValueError:
            await self._set_state(self._idle_state())
            return
        self._turn_counter += 1
        metrics = TurnMetrics(
            turn_id=self._turn_counter,
            speech_start=ev.speech_start,
            speech_end=ev.speech_end,
            turn_final=ev.final_at,
            asr_finalize_ms=ev.asr_finalize_ms,
        )
        await self._send({"type": "transcript", "final": True, "text": ev.text,
                          "turn_id": self._turn_counter, "user_turn_text": turn.content})
        await self._start_response(metrics)

    async def _start_response(self, metrics: TurnMetrics) -> None:
        if self._services.tts is None:
            await self._error("tts_unavailable", "Text-to-speech is not available", stage="tts")
            await self._set_state(self._idle_state())
            return
        self._generation += 1
        gen = self._generation
        self._active_gen = gen
        self._metrics[gen] = metrics
        self._conv.start_assistant(gen)
        messages = self._conv.build_messages(self._services.system_prompt)
        await self._set_state(AssistantState.THINKING)
        await self._send({"type": "assistant_start", "generation_id": gen, "turn_id": metrics.turn_id})
        self._response_task = asyncio.create_task(self._run_response(gen, messages, metrics), name=f"resp-{gen}")

    async def _run_response(self, gen: int, messages, metrics: TurnMetrics) -> None:
        assert self._services.tts is not None
        streamer = ResponseStreamer(
            llm=self._services.llm,
            tts=self._services.tts,
            sink=self,
            chunking=ChunkerSettings(
                self._s.chunk_min_chars, self._s.chunk_max_chars, self._s.chunk_first_soft_chars, self._s.chunk_soft_chars
            ),
            voice=self._voice,
            speed=self._s.tts_speed,
        )
        try:
            result = await streamer.run(gen, messages, metrics)
        except asyncio.CancelledError:
            raise
        except VoiceAssistantError as exc:
            result = ResponseResult("", 0, exc)
        except Exception as exc:  # defensive: never let a response crash the session
            log.exception("Response failed", extra={"session": self.id, "generation": gen})
            result = ResponseResult("", metrics.sentences, VoiceAssistantError(f"{type(exc).__name__}"))
        if gen != self._active_gen:
            return  # interrupted while finishing
        self._response_done.add(gen)
        if result.error is not None:
            metrics.error = result.error.code
            await self._error(result.error.code, str(result.error) or "Response failed", stage=result.error.stage)
        await self._send({"type": "assistant_end", "generation_id": gen, "sentences": result.sentences})
        if result.sentences == 0:
            await self._complete_generation(gen, interrupted=False)
        # Otherwise the turn completes when the client reports that playback drained.

    # ResponseSink ------------------------------------------------------------------------------

    async def on_sentence(self, generation_id: int, seq: int, text: str) -> None:
        if generation_id != self._active_gen:
            return
        self._conv.record_sentence(generation_id, seq, text)
        await self._send({"type": "assistant_text", "generation_id": generation_id, "seq": seq, "text": text})

    async def on_audio(self, generation_id: int, seq: int, audio: SynthesizedAudio) -> None:
        if generation_id != self._active_gen:
            return  # stale audio from an interrupted generation is never sent
        try:
            await self._transport.send_audio(generation_id, seq, audio)
        except Exception:
            log.debug("send_audio failed (client gone?)", extra={"session": self.id})

    # --- interruption -------------------------------------------------------------------------

    async def interrupt(self, reason: str) -> bool:
        gen = self._active_gen
        if gen is None:
            return False
        self._active_gen = None  # stale guard first, then cancel
        task = self._response_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=2.0)
        self._interruptions += 1
        metrics = self._metrics.get(gen)
        if metrics is not None:
            metrics.interrupted = True
        await self._send({"type": "interrupted", "generation_id": gen, "reason": reason})
        await self._complete_generation(gen, interrupted=True)
        log.info("Interrupted", extra={"session": self.id, "generation": gen, "reason": reason})
        return True

    async def _complete_generation(self, gen: int, interrupted: bool) -> None:
        turn = self._conv.finish_assistant(gen, interrupted)
        self._response_done.discard(gen)
        metrics = self._metrics.pop(gen, None)
        if metrics is not None and metrics.response_done is None:
            metrics.response_done = now()
        if self._active_gen == gen:
            self._active_gen = None
        await self._send({"type": "assistant_final", "generation_id": gen, "interrupted": interrupted,
                          "text": turn.content if turn else ""})
        if metrics is not None:
            await self._send({"type": "metrics", "turn": metrics.to_dict(),
                              "session": {"turns": self._turn_counter, "interruptions": self._interruptions,
                                          "dropped_audio_frames": self._dropped_frames}})
        await self._set_state(self._idle_state())

    # --- client control messages --------------------------------------------------------------

    async def handle_control(self, msg: dict[str, Any]) -> None:
        kind = msg["type"]
        if kind == "start":
            if self._pipeline is None:
                await self._error("speech_unavailable", "Speech models are not loaded; see server log", stage="models")
                return
            self._listening = True
            if self._active_gen is None:
                await self._set_state(AssistantState.LISTENING)
        elif kind == "stop":
            self._listening = False
            await self.interrupt("mic_stopped")
            await self._queue_reset()
            await self._set_state(AssistantState.IDLE)
        elif kind == "interrupt":
            await self.interrupt("manual")
        elif kind == "playback":
            await self._on_playback(msg)
        elif kind == "reset":
            await self.interrupt("reset")
            await self._queue_reset()
            self._conv.reset()
            await self._send({"type": "history_reset"})
        elif kind == "set_voice":
            voice = msg["voice"]
            tts = self._services.tts
            if tts is None or voice not in tts.voices():
                await self._error("bad_voice", f"Unknown voice '{voice}'", stage="tts")
                return
            self._voice = voice  # applies from the next response; keeps a reply in one voice
            await self._send({"type": "voice_changed", "voice": voice})
        elif kind == "ping":
            await self._send({"type": "pong", "t": msg.get("t")})

    async def _on_playback(self, msg: dict[str, Any]) -> None:
        gen = msg["generation_id"]
        if gen != self._active_gen:
            return  # notice for an interrupted/finished generation
        if msg["event"] == "started":
            self._conv.record_played(gen, msg["seq"])
            metrics = self._metrics.get(gen)
            if metrics is not None and metrics.first_playback is None:
                metrics.first_playback = now()
            await self._set_state(AssistantState.SPEAKING)
        elif msg["event"] == "idle" and gen in self._response_done:
            await self._complete_generation(gen, interrupted=False)

    async def _queue_reset(self) -> None:
        while not self._audio_q.empty():
            self._audio_q.get_nowait()
        if self._pipeline is not None:
            self._audio_q.put_nowait(_RESET)

    # --- helpers ------------------------------------------------------------------------------

    def _idle_state(self) -> AssistantState:
        if not self._listening:
            return AssistantState.IDLE
        if self._pipeline is not None and self._pipeline.in_utterance:
            return AssistantState.USER_SPEAKING
        return AssistantState.LISTENING

    async def _set_state(self, state: AssistantState) -> None:
        if state is not self._state:
            self._state = state
            await self._send({"type": "state", "state": state.value})

    async def _error(self, code: str, message: str, stage: str) -> None:
        await self._send({"type": "error", "code": code, "message": message, "stage": stage})

    async def _send(self, event: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            await self._transport.send_event(event)
        except Exception:
            log.debug("send_event failed (client gone?)", extra={"session": self.id, "type": event.get("type")})

    @property
    def state(self) -> AssistantState:
        return self._state

    @property
    def conversation(self) -> ConversationManager:
        return self._conv
