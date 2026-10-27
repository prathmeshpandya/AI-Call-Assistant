"""Voice activity detection behind a small interface.

The Silero implementation uses sherpa-onnx's built-in VAD, so no PyTorch dependency is needed.
`is_speech()` reports whether the detector is currently inside a speech segment; turn-level
decisions (pauses vs. end of turn) are made by core.turn_detector, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app.errors import ModelNotFoundError


class VoiceActivityDetector(ABC):
    window_size: int  # samples per decision window

    @abstractmethod
    def accept(self, window: np.ndarray) -> bool:
        """Feed exactly `window_size` float32 samples; return True while speech is ongoing."""

    @abstractmethod
    def reset(self) -> None: ...


class SileroVAD(VoiceActivityDetector):
    def __init__(
        self,
        model_path: Path,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ms: int = 200,
        min_silence_ms: int = 150,
    ) -> None:
        if not model_path.is_file():
            raise ModelNotFoundError(f"Silero VAD model not found at {model_path}. Run scripts\\download-models.ps1")
        import sherpa_onnx  # imported lazily so unit tests do not need the native library

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = str(model_path)
        config.silero_vad.threshold = threshold
        config.silero_vad.min_speech_duration = min_speech_ms / 1000.0
        config.silero_vad.min_silence_duration = min_silence_ms / 1000.0
        config.silero_vad.max_speech_duration = 60.0
        config.sample_rate = sample_rate
        config.num_threads = 1
        self._config = config
        self.window_size = config.silero_vad.window_size  # 512 samples (32 ms) for Silero at 16 kHz
        self._vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=10)

    def accept(self, window: np.ndarray) -> bool:
        self._vad.accept_waveform(window)
        # We only need the live speech flag; drop completed segments so the internal buffer stays small.
        while not self._vad.empty():
            self._vad.pop()
        return bool(self._vad.is_speech_detected())

    def reset(self) -> None:
        self._vad.reset()
