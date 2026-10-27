"""Text-to-speech interface. Swap engines (e.g. a more expressive model) without touching the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SynthesizedAudio:
    samples: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int


class TTSEngine(ABC):
    name: str
    sample_rate: int

    @abstractmethod
    def voices(self) -> list[str]: ...

    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> SynthesizedAudio:
        """Blocking synthesis of one sentence. Callers run it in a worker thread."""

    def close(self) -> None:  # pragma: no cover - default no-op
        """Release native resources."""
