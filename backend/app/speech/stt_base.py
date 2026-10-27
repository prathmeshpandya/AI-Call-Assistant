"""Speech-recognition interface.

An engine is loaded once (heavy, shared); each conversation gets its own lightweight stream.
Implementations must be honest about streaming: `partial()` is only meaningful for engines that
decode incrementally. A batch engine (e.g. Whisper) would return "" from partial() and do all of
its work in finalize().
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class STTStream(ABC):
    @abstractmethod
    def accept(self, samples: np.ndarray) -> None:
        """Feed float32 mono samples at the engine's sample rate. Decodes incrementally if supported."""

    @abstractmethod
    def partial(self) -> str:
        """Current provisional hypothesis (may change). Never stored in conversation history."""

    @abstractmethod
    def finalize(self) -> str:
        """Flush remaining audio and return the final transcript for the utterance."""

    @abstractmethod
    def reset(self) -> None:
        """Discard state and start a new utterance."""


class STTEngine(ABC):
    name: str
    sample_rate: int
    is_streaming: bool

    @abstractmethod
    def create_stream(self) -> STTStream: ...

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release native resources."""
