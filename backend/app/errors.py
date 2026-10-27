"""Domain exceptions shared across modules (kept dependency-free to avoid import cycles)."""


class VoiceAssistantError(Exception):
    """Base error. `code` is a stable identifier sent to the client; `stage` names the pipeline stage."""

    code = "internal_error"
    stage = "backend"


class ModelNotFoundError(VoiceAssistantError):
    code = "model_not_found"
    stage = "models"


class STTError(VoiceAssistantError):
    code = "stt_failed"
    stage = "stt"


class TTSError(VoiceAssistantError):
    code = "tts_failed"
    stage = "tts"


class LLMUnavailableError(VoiceAssistantError):
    code = "llm_unavailable"
    stage = "llm"


class LLMError(VoiceAssistantError):
    code = "llm_failed"
    stage = "llm"
