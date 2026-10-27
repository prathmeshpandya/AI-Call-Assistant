"""Application configuration loaded from environment variables (and the project-root .env file).

Every tunable lives here so business logic never reads os.environ directly.
Relative paths are resolved against the project root, so the backend can be started from any cwd.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


def _csv(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Server ---
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    # Origins allowed to open the WebSocket (protects the local mic/LLM backend from arbitrary websites).
    frontend_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"
    log_json: bool = False
    max_sessions: int = 4
    warmup: bool = True

    # --- Language model (OpenAI-compatible, e.g. llama.cpp llama-server) ---
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str = "local-slm"
    llm_api_key: str = ""  # llama-server needs none; only set for an explicitly configured remote provider
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_max_tokens: int = 300
    llm_connect_timeout_s: float = 3.0
    llm_first_token_timeout_s: float = 30.0
    llm_total_timeout_s: float = 120.0
    system_prompt_file: Path = BACKEND_DIR / "prompts" / "system_prompt.txt"

    # --- Language ---
    language: str = "en"

    # --- Speech recognition (sherpa-onnx streaming transducer) ---
    stt_provider: str = "sherpa_streaming"
    stt_model_dir: Path = Path("models/sherpa-onnx-streaming-zipformer-en-2023-06-26")
    # Optional explicit file overrides; otherwise discovered inside stt_model_dir.
    stt_encoder: Path | None = None
    stt_decoder: Path | None = None
    stt_joiner: Path | None = None
    stt_tokens: Path | None = None
    stt_num_threads: int = 2
    stt_decoding_method: str = "greedy_search"
    stt_tail_padding_ms: int = 500

    # --- Voice activity detection & turn-taking ---
    vad_model_path: Path = Path("models/silero_vad.onnx")
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 200
    vad_min_silence_ms: int = 150
    preroll_ms: int = 500
    end_of_turn_silence_ms: int = 700
    end_of_turn_hesitation_silence_ms: int = 1500
    min_utterance_ms: int = 250
    max_utterance_s: float = 30.0

    # --- Barge-in (user interrupting the assistant) ---
    barge_in_enabled: bool = True
    barge_in_min_speech_ms: int = 400
    barge_in_min_words: int = 2
    barge_in_keywords: str = "stop,wait,hold on,no,sorry,excuse me,hang on"

    # --- Text-to-speech (sherpa-onnx) ---
    # "piper_sherpa" (default): ~0.1x real-time on a Ryzen 5 5500U even while the LLM generates.
    # "kokoro_sherpa": more natural, but measured slower than real-time on that CPU under LLM load,
    # which causes audible gaps between sentences. Use it on faster CPUs or with a GPU-offloaded LLM.
    tts_provider: str = "piper_sherpa"
    tts_model_dir: Path = Path("models/vits-piper-en_US-ljspeech-medium")
    tts_voice: str = ""  # empty = the engine's first voice; Kokoro example: af_bella
    tts_speed: float = 1.0
    tts_num_threads: int = 2

    # --- Audio transport ---
    audio_input_sample_rate: int = 16000
    audio_channels: int = 1
    audio_max_frame_bytes: int = 32000  # 1 s of 16 kHz PCM16 per message at most

    # --- Sentence chunking for streaming TTS ---
    chunk_min_chars: int = 12
    chunk_max_chars: int = 220
    chunk_first_soft_chars: int = 30  # first chunk may break at a clause (, ; : dash) once this long
    chunk_soft_chars: int = 40  # later chunks too; short chunks keep TTS ahead of playback

    # --- Conversation memory ---
    max_history_turns: int = 12
    max_history_chars: int = 6000

    @field_validator("llm_base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("audio_channels")
    @classmethod
    def _mono_only(cls, v: int) -> int:
        if v != 1:
            raise ValueError("Only mono input (AUDIO_CHANNELS=1) is supported")
        return v

    @field_validator("audio_input_sample_rate")
    @classmethod
    def _rate(cls, v: int) -> int:
        if v != 16000:
            raise ValueError("AUDIO_INPUT_SAMPLE_RATE must be 16000 (required by Silero VAD and the ASR model)")
        return v

    @property
    def allowed_origins(self) -> list[str]:
        return _csv(self.frontend_origins)

    @property
    def barge_in_keyword_list(self) -> list[str]:
        return [k.lower() for k in _csv(self.barge_in_keywords)]

    def resolve(self, path: Path | None) -> Path | None:
        """Resolve a configured path relative to the project root."""
        if path is None:
            return None
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    def load_system_prompt(self) -> str:
        path = self.resolve(self.system_prompt_file)
        assert path is not None
        return path.read_text(encoding="utf-8").strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings", "PROJECT_ROOT", "BACKEND_DIR", "Field"]
