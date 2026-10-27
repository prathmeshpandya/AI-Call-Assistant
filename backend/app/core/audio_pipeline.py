"""Per-session input pipeline: PCM samples -> VAD -> turn detection -> streaming STT -> events.

Synchronous and CPU-bound by design; the session runs `process()` in a worker thread so the event
loop stays responsive. Transport-agnostic: it never touches WebSockets.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from app.audio.vad import VoiceActivityDetector
from app.core.metrics import now
from app.core.turn_detector import TurnConfig, TurnDetector, TurnEvent, TurnState
from app.errors import STTError
from app.speech.stt_base import STTStream

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechStarted:
    at: float


@dataclass(frozen=True)
class PartialTranscript:
    text: str
    speech_ms: float


@dataclass(frozen=True)
class FinalTranscript:
    text: str
    speech_start: float
    speech_end: float
    final_at: float
    asr_finalize_ms: float
    speech_ms: float


@dataclass(frozen=True)
class UtteranceDiscarded:
    reason: str


@dataclass(frozen=True)
class PipelineError:
    message: str


PipelineEvent = SpeechStarted | PartialTranscript | FinalTranscript | UtteranceDiscarded | PipelineError


class AudioPipeline:
    def __init__(
        self,
        vad: VoiceActivityDetector,
        stt: STTStream,
        turn_config: TurnConfig,
        sample_rate: int = 16000,
        preroll_ms: int = 500,
        vad_trigger_delay_ms: int = 0,
    ) -> None:
        self._onset_offset_s = vad_trigger_delay_ms / 1000.0
        self._vad = vad
        self._stt = stt
        self._turns = TurnDetector(turn_config)
        self._window = vad.window_size
        self._frame_ms = 1000.0 * self._window / sample_rate
        # Keep audio from just before VAD triggers so the first syllable reaches the recognizer.
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, int(preroll_ms / self._frame_ms)))
        self._residual = np.zeros(0, dtype=np.float32)
        self._partial = ""
        self._speech_start = 0.0
        self._last_voiced = 0.0

    @property
    def in_utterance(self) -> bool:
        return self._turns.state is not TurnState.IDLE

    @property
    def speech_ms(self) -> float:
        return self._turns.speech_ms

    @property
    def partial(self) -> str:
        return self._partial

    def reset(self) -> None:
        self._turns.reset()
        self._vad.reset()
        self._stt.reset()
        self._preroll.clear()
        self._residual = np.zeros(0, dtype=np.float32)
        self._partial = ""

    def process(self, samples: np.ndarray) -> list[PipelineEvent]:
        events: list[PipelineEvent] = []
        buf = np.concatenate([self._residual, samples]) if self._residual.size else samples
        n_full = len(buf) // self._window * self._window
        self._residual = buf[n_full:].copy()
        for start in range(0, n_full, self._window):
            try:
                self._process_window(buf[start : start + self._window], events)
            except STTError as exc:
                log.exception("STT failure; dropping utterance")
                self._stt.reset()
                self._turns.reset()
                self._partial = ""
                events.append(PipelineError(str(exc)))
        return events

    def _process_window(self, window: np.ndarray, events: list[PipelineEvent]) -> None:
        is_speech = self._vad.accept(window)
        was_idle = self._turns.state is TurnState.IDLE
        event = self._turns.process(is_speech, self._frame_ms, self._partial)
        t = now()
        if is_speech:
            self._last_voiced = t

        if was_idle and event is not TurnEvent.SPEECH_START:
            self._preroll.append(window)
            return

        if event is TurnEvent.SPEECH_START:
            # VAD needs ~min_speech_ms of voice to trigger, so the real onset was that much earlier.
            self._speech_start = t - self._onset_offset_s
            for w in self._preroll:
                self._stt.accept(w)
            self._preroll.clear()
            self._stt.accept(window)
            events.append(SpeechStarted(at=t))
            return

        self._stt.accept(window)
        partial = self._stt.partial()
        if partial != self._partial:
            self._partial = partial
            events.append(PartialTranscript(partial, self._turns.speech_ms))

        if event is TurnEvent.END_OF_TURN:
            speech_ms = self._turns.speech_ms
            t0 = now()
            text = self._stt.finalize()
            t1 = now()
            self._stt.reset()
            self._partial = ""
            if text.strip():
                events.append(
                    FinalTranscript(
                        text=text,
                        speech_start=self._speech_start,
                        speech_end=self._last_voiced,
                        final_at=t1,
                        asr_finalize_ms=round((t1 - t0) * 1000.0, 1),
                        speech_ms=speech_ms,
                    )
                )
            else:
                events.append(UtteranceDiscarded("no words recognised"))
        elif event is TurnEvent.DISCARD:
            self._stt.reset()
            self._partial = ""
            events.append(UtteranceDiscarded("too short"))
