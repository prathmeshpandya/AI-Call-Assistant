"""Turn-taking: decides when the user started speaking, paused, and finished their turn.

Pure state machine driven by per-window VAD decisions, so it is deterministic and unit-testable.

    IDLE --speech--> SPEAKING --silence--> PAUSED --speech--> SPEAKING
                                              |
                                              +-- silence >= threshold --> END_OF_TURN (or DISCARD if too short)

The silence threshold is longer when the partial transcript ends in a word that suggests the user
is mid-thought ("and", "so", "because", ...), which avoids answering every thinking pause.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Words that almost never end a finished utterance. Prepositions are deliberately excluded:
# English questions often end in them ("what is it famous for?"), which measurably delayed replies.
_HESITATION_TAIL = {
    "and", "but", "so", "or", "because", "um", "uh", "uhm", "er", "erm", "hmm",
    "the", "a", "an", "my", "your", "our", "their", "if", "then",
}


class TurnState(str, Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    PAUSED = "paused"


class TurnEvent(str, Enum):
    SPEECH_START = "speech_start"
    END_OF_TURN = "end_of_turn"
    DISCARD = "discard"  # too short to be intentional speech (cough, click, noise)


@dataclass(frozen=True)
class TurnConfig:
    end_of_turn_silence_ms: int = 700
    hesitation_silence_ms: int = 1500
    min_utterance_ms: int = 250
    max_utterance_ms: int = 30_000


def ends_with_hesitation(transcript: str) -> bool:
    words = transcript.lower().split()
    return bool(words) and words[-1].strip(",.?!") in _HESITATION_TAIL


class TurnDetector:
    def __init__(self, config: TurnConfig) -> None:
        self.cfg = config
        self.state = TurnState.IDLE
        self.speech_ms = 0.0  # voiced time in the current utterance
        self.silence_ms = 0.0  # trailing silence in the current pause
        self.total_ms = 0.0

    def reset(self) -> None:
        self.state = TurnState.IDLE
        self.speech_ms = self.silence_ms = self.total_ms = 0.0

    def required_silence_ms(self, transcript: str) -> int:
        if ends_with_hesitation(transcript):
            return self.cfg.hesitation_silence_ms
        return self.cfg.end_of_turn_silence_ms

    def process(self, is_speech: bool, frame_ms: float, transcript: str = "") -> TurnEvent | None:
        if self.state is TurnState.IDLE:
            if not is_speech:
                return None
            self.state = TurnState.SPEAKING
            self.speech_ms, self.silence_ms, self.total_ms = frame_ms, 0.0, frame_ms
            return TurnEvent.SPEECH_START

        self.total_ms += frame_ms
        if is_speech:
            self.state = TurnState.SPEAKING
            self.speech_ms += frame_ms
            self.silence_ms = 0.0
        else:
            self.state = TurnState.PAUSED
            self.silence_ms += frame_ms

        if self.total_ms >= self.cfg.max_utterance_ms:
            return self._finish(TurnEvent.END_OF_TURN)
        if self.state is TurnState.PAUSED and self.silence_ms >= self.required_silence_ms(transcript):
            if self.speech_ms < self.cfg.min_utterance_ms:
                return self._finish(TurnEvent.DISCARD)
            return self._finish(TurnEvent.END_OF_TURN)
        return None

    def _finish(self, event: TurnEvent) -> TurnEvent:
        self.state = TurnState.IDLE
        return event
