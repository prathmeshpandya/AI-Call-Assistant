"""Per-turn latency measurement. All timestamps come from time.perf_counter() on the server, so
deltas are consistent. 'first_playback' is the server-side receipt time of the client's
"playback started" notice, i.e. includes browser scheduling plus one local network hop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now() -> float:
    return time.perf_counter()


def _ms(a: float | None, b: float | None) -> float | None:
    return round((b - a) * 1000.0, 1) if a is not None and b is not None else None


@dataclass
class TurnMetrics:
    turn_id: int
    speech_start: float | None = None
    speech_end: float | None = None  # last voiced audio of the utterance
    turn_final: float | None = None  # final transcript available
    asr_finalize_ms: float | None = None
    llm_request: float | None = None
    first_token: float | None = None
    first_sentence: float | None = None
    first_audio_sent: float | None = None
    first_playback: float | None = None
    response_done: float | None = None
    tts_ms: list[float] = field(default_factory=list)
    audio_ms: list[float] = field(default_factory=list)
    sentences: int = 0
    tokens: int = 0
    interrupted: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "speech_duration_ms": _ms(self.speech_start, self.speech_end),
            "speech_start_to_final_ms": _ms(self.speech_start, self.turn_final),
            "end_of_speech_to_final_ms": _ms(self.speech_end, self.turn_final),
            "asr_finalize_ms": self.asr_finalize_ms,
            "final_to_first_token_ms": _ms(self.turn_final, self.first_token),
            "final_to_first_sentence_ms": _ms(self.turn_final, self.first_sentence),
            "final_to_first_audio_ms": _ms(self.turn_final, self.first_audio_sent),
            "final_to_first_playback_ms": _ms(self.turn_final, self.first_playback),
            "end_of_speech_to_first_playback_ms": _ms(self.speech_end, self.first_playback),
            "tts_ms": [round(x, 1) for x in self.tts_ms],
            "tts_total_ms": round(sum(self.tts_ms), 1),
            "audio_total_ms": round(sum(self.audio_ms), 1),
            "response_total_ms": _ms(self.turn_final, self.response_done),
            "sentences": self.sentences,
            "tokens": self.tokens,
            "interrupted": self.interrupted,
            "error": self.error,
        }
