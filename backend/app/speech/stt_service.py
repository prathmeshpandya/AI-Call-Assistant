"""sherpa-onnx streaming transducer (Zipformer) speech recognition.

This is true streaming recognition: audio is decoded chunk by chunk as it arrives, so partial
hypotheses are available while the user is still speaking. End-of-turn is decided by our own
VAD-based turn detector, so sherpa's built-in endpoint rules are disabled.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from app.config import Settings
from app.errors import ModelNotFoundError, STTError
from app.speech.stt_base import STTEngine, STTStream

log = logging.getLogger(__name__)


def _pick(model_dir: Path, prefix: str, prefer_int8: bool) -> Path:
    candidates = sorted(model_dir.glob(f"{prefix}*.onnx"))
    if not candidates:
        raise ModelNotFoundError(f"No {prefix}*.onnx in {model_dir}")
    int8 = [c for c in candidates if c.name.endswith(".int8.onnx")]
    fp32 = [c for c in candidates if not c.name.endswith(".int8.onnx")]
    ordered = (int8 + fp32) if prefer_int8 else (fp32 + int8)
    return ordered[0]


class SherpaStreamingSTT(STTEngine):
    name = "sherpa-onnx streaming zipformer"
    is_streaming = True

    def __init__(self, settings: Settings) -> None:
        import sherpa_onnx

        model_dir = settings.resolve(settings.stt_model_dir)
        assert model_dir is not None
        if not model_dir.is_dir():
            raise ModelNotFoundError(
                f"STT model directory not found: {model_dir}. Run scripts\\download-models.ps1"
            )
        # int8 encoder/joiner are ~4x smaller and faster on CPU with negligible accuracy loss;
        # the decoder is tiny, so the fp32 one is used.
        encoder = settings.resolve(settings.stt_encoder) or _pick(model_dir, "encoder", prefer_int8=True)
        decoder = settings.resolve(settings.stt_decoder) or _pick(model_dir, "decoder", prefer_int8=False)
        joiner = settings.resolve(settings.stt_joiner) or _pick(model_dir, "joiner", prefer_int8=True)
        tokens = settings.resolve(settings.stt_tokens) or model_dir / "tokens.txt"
        for f in (encoder, decoder, joiner, tokens):
            if not f.is_file():
                raise ModelNotFoundError(f"STT model file missing: {f}")

        self.sample_rate = settings.audio_input_sample_rate
        self._tail = np.zeros(int(self.sample_rate * settings.stt_tail_padding_ms / 1000), dtype=np.float32)
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=settings.stt_num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method=settings.stt_decoding_method,
            enable_endpoint_detection=False,
            provider="cpu",
        )
        # The recognizer is shared across sessions; serialize decode calls to be safe.
        self._lock = threading.Lock()
        log.info("STT loaded", extra={"encoder": encoder.name, "threads": settings.stt_num_threads})

    def create_stream(self) -> STTStream:
        return _SherpaStream(self)


class _SherpaStream(STTStream):
    def __init__(self, engine: SherpaStreamingSTT) -> None:
        self._engine = engine
        self._stream = engine._recognizer.create_stream()

    def _decode(self) -> None:
        rec = self._engine._recognizer
        with self._engine._lock:
            while rec.is_ready(self._stream):
                rec.decode_stream(self._stream)

    def accept(self, samples: np.ndarray) -> None:
        try:
            self._stream.accept_waveform(self._engine.sample_rate, samples)
            self._decode()
        except Exception as exc:  # native errors surface as RuntimeError
            raise STTError(f"decode failed: {exc}") from exc

    def partial(self) -> str:
        return _normalize(self._engine._recognizer.get_result(self._stream))

    def finalize(self) -> str:
        try:
            # Tail padding flushes the model's right context so the last word is emitted.
            self._stream.accept_waveform(self._engine.sample_rate, self._engine._tail)
            self._stream.input_finished()
            self._decode()
            return self.partial()
        except Exception as exc:
            raise STTError(f"finalize failed: {exc}") from exc

    def reset(self) -> None:
        self._stream = self._engine._recognizer.create_stream()


def _normalize(text: object) -> str:
    # get_result returns str in current sherpa-onnx; older builds returned an object with .text
    raw = text if isinstance(text, str) else getattr(text, "text", "")
    # Zipformer English models emit upper-case text without punctuation.
    cleaned = " ".join(raw.split()).lower()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else ""


def create_stt_engine(settings: Settings) -> STTEngine:
    if settings.stt_provider == "sherpa_streaming":
        return SherpaStreamingSTT(settings)
    raise ValueError(f"Unknown STT_PROVIDER '{settings.stt_provider}'")
