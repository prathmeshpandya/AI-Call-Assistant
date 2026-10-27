"""Model/service container: loads heavy engines once at startup and reports their health.

A missing model does not crash the server; the component is marked unavailable and every client
session is told exactly what is missing and how to fix it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from app.audio.vad import SileroVAD, VoiceActivityDetector
from app.config import Settings
from app.errors import ModelNotFoundError, VoiceAssistantError
from app.llm.llm_base import ChatMessage, LLMClient
from app.llm.llm_service import create_llm_client
from app.speech.stt_base import STTEngine
from app.speech.stt_service import create_stt_engine
from app.tts.tts_base import TTSEngine
from app.tts.tts_service import create_tts_engine

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    llm: LLMClient
    system_prompt: str
    stt: STTEngine | None = None
    tts: TTSEngine | None = None
    errors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, settings: Settings) -> "Services":
        errors: dict[str, str] = {}
        stt = tts = None
        try:
            stt = create_stt_engine(settings)
        except (ModelNotFoundError, VoiceAssistantError, ValueError) as exc:
            errors["stt"] = str(exc)
        try:
            tts = create_tts_engine(settings)
        except (ModelNotFoundError, VoiceAssistantError, ValueError) as exc:
            errors["tts"] = str(exc)
        vad_path = settings.resolve(settings.vad_model_path)
        if vad_path is None or not vad_path.is_file():
            errors["vad"] = f"Silero VAD model not found at {vad_path}. Run scripts\\download-models.ps1"
        for component, msg in errors.items():
            log.error("Component unavailable", extra={"component": component, "detail": msg})
        return cls(
            settings=settings,
            llm=create_llm_client(settings),
            system_prompt=settings.load_system_prompt(),
            stt=stt,
            tts=tts,
            errors=errors,
        )

    @property
    def speech_ready(self) -> bool:
        return self.stt is not None and self.tts is not None and "vad" not in self.errors

    def create_vad(self) -> VoiceActivityDetector:
        s = self.settings
        path = s.resolve(s.vad_model_path)
        assert path is not None
        return SileroVAD(
            path,
            sample_rate=s.audio_input_sample_rate,
            threshold=s.vad_threshold,
            min_speech_ms=s.vad_min_speech_ms,
            min_silence_ms=s.vad_min_silence_ms,
        )

    def default_voice(self) -> str:
        if self.tts is None:
            return self.settings.tts_voice
        voices = self.tts.voices()
        if self.settings.tts_voice in voices:
            return self.settings.tts_voice
        if not self.settings.tts_voice:
            return voices[0]
        log.warning("Configured TTS_VOICE not available; using first voice",
                    extra={"configured": self.settings.tts_voice, "using": voices[0]})
        return voices[0]

    async def warmup(self) -> None:
        """First inference is slow (graph init, page-faulting weights, empty prompt cache)."""
        if self.tts is not None:
            t = asyncio.get_running_loop().time()
            await asyncio.to_thread(self.tts.synthesize, "Hello there.", self.default_voice())
            log.info("TTS warm", extra={"ms": round((asyncio.get_running_loop().time() - t) * 1000)})
        health = await self.llm.health()
        if not health.ok:
            log.warning("LLM not reachable during warm-up", extra={"detail": health.detail})
            return
        try:
            messages = [ChatMessage("system", self.system_prompt), ChatMessage("user", "Hi")]
            stream = self.llm.stream_chat(messages)
            try:
                await anext(stream)  # one token is enough to load weights and cache the system prompt
            except StopAsyncIteration:
                pass
            finally:
                await stream.aclose()  # closes the HTTP stream so the server stops generating
            log.info("LLM warm")
        except VoiceAssistantError as exc:
            log.warning("LLM warm-up failed", extra={"error": str(exc)})

    async def aclose(self) -> None:
        await self.llm.aclose()
        for engine in (self.stt, self.tts):
            if engine is not None:
                engine.close()
