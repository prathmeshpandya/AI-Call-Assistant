"""Offline TTS engines via sherpa-onnx (bundled espeak-ng data, no extra system installs).

- KokoroSherpaTTS: Kokoro-82M (Apache-2.0). Best quality; measured ~0.77x real-time on a Ryzen 5 5500U (fp32, 4 threads).
- PiperSherpaTTS:  Piper/VITS voices. ~0.08x real-time on the same CPU; voice licenses vary per dataset.

Neither engine streams audio token-by-token, so the response streamer synthesizes one sentence at a
time and overlaps that with LLM generation and with playback of the previous sentence.

Note: sherpa-onnx 2.x removed built-in espeak-ng phonemization, which these models rely on, so
requirements.txt pins sherpa-onnx 1.12.x.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from app.config import Settings
from app.errors import ModelNotFoundError, TTSError
from app.tts.tts_base import SynthesizedAudio, TTSEngine

log = logging.getLogger(__name__)

# Speaker order of kokoro-en-v0_19 (see https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html)
KOKORO_EN_V019_VOICES = [
    "af", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
]


def _model_dir(settings: Settings) -> Path:
    model_dir = settings.resolve(settings.tts_model_dir)
    assert model_dir is not None
    if not model_dir.is_dir():
        raise ModelNotFoundError(f"TTS model directory not found: {model_dir}. Run scripts\\download-models.ps1")
    return model_dir


def _require(model_dir: Path, *names: str) -> None:
    for name in names:
        if not (model_dir / name).exists():
            raise ModelNotFoundError(f"TTS model file missing: {model_dir / name}")


def _find_onnx(model_dir: Path, preferred: tuple[str, ...] = ()) -> Path:
    for name in preferred:
        if (model_dir / name).is_file():
            return model_dir / name
    found = sorted(model_dir.glob("*.onnx"))
    if not found:
        raise ModelNotFoundError(f"No .onnx TTS model in {model_dir}")
    return found[0]


class _SherpaTTS(TTSEngine):
    """Shared synthesis logic for sherpa-onnx OfflineTts models."""

    def _init_engine(self, model_config, voices_for: callable) -> None:  # type: ignore[valid-type]
        import sherpa_onnx

        config = sherpa_onnx.OfflineTtsConfig(model=model_config, max_num_sentences=1)
        if not config.validate():
            raise TTSError(f"Invalid {self.name} configuration")
        self._tts = sherpa_onnx.OfflineTts(config)
        self.sample_rate = int(self._tts.sample_rate)
        self._voices: list[str] = voices_for(int(self._tts.num_speakers))
        self._lock = threading.Lock()  # one synthesis at a time on the shared native object

    def voices(self) -> list[str]:
        return list(self._voices)

    def _speaker_id(self, voice: str) -> int:
        if voice in self._voices:
            return self._voices.index(voice)
        if voice.isdigit() and int(voice) < len(self._voices):
            return int(voice)
        raise TTSError(f"Unknown voice '{voice}'. Available: {', '.join(self._voices)}")

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> SynthesizedAudio:
        sid = self._speaker_id(voice)
        try:
            with self._lock:
                audio = self._tts.generate(text, sid=sid, speed=speed)
        except Exception as exc:
            raise TTSError(f"synthesis failed: {exc}") from exc
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            raise TTSError("synthesis produced no audio")
        return SynthesizedAudio(samples=samples, sample_rate=int(audio.sample_rate))


class KokoroSherpaTTS(_SherpaTTS):
    name = "kokoro (sherpa-onnx)"

    def __init__(self, settings: Settings) -> None:
        import sherpa_onnx

        model_dir = _model_dir(settings)
        _require(model_dir, "voices.bin", "tokens.txt", "espeak-ng-data")
        model = _find_onnx(model_dir, ("model.onnx", "model.int8.onnx"))
        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(model),
            voices=str(model_dir / "voices.bin"),
            tokens=str(model_dir / "tokens.txt"),
            data_dir=str(model_dir / "espeak-ng-data"),
        )
        lexicons = sorted(model_dir.glob("lexicon*.txt"))  # multi-lingual Kokoro releases ship these
        if lexicons:
            kokoro.lexicon = ",".join(str(p) for p in lexicons)
        self._init_engine(
            sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=settings.tts_num_threads, provider="cpu"),
            lambda n: KOKORO_EN_V019_VOICES if n == len(KOKORO_EN_V019_VOICES) else [f"speaker_{i}" for i in range(n)],
        )
        log.info("TTS loaded", extra={"engine": self.name, "model": model.name, "sample_rate": self.sample_rate})


class PiperSherpaTTS(_SherpaTTS):
    name = "piper/vits (sherpa-onnx)"

    def __init__(self, settings: Settings) -> None:
        import sherpa_onnx

        model_dir = _model_dir(settings)
        _require(model_dir, "tokens.txt", "espeak-ng-data")
        model = _find_onnx(model_dir)
        voice_name = model.stem
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(model), tokens=str(model_dir / "tokens.txt"), data_dir=str(model_dir / "espeak-ng-data")
        )
        self._init_engine(
            sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=settings.tts_num_threads, provider="cpu"),
            lambda n: [voice_name] if n == 1 else [f"{voice_name}#{i}" for i in range(n)],
        )
        log.info("TTS loaded", extra={"engine": self.name, "model": model.name, "sample_rate": self.sample_rate})


def create_tts_engine(settings: Settings) -> TTSEngine:
    if settings.tts_provider == "kokoro_sherpa":
        return KokoroSherpaTTS(settings)
    if settings.tts_provider == "piper_sherpa":
        return PiperSherpaTTS(settings)
    raise ValueError(f"Unknown TTS_PROVIDER '{settings.tts_provider}' (expected kokoro_sherpa or piper_sherpa)")
