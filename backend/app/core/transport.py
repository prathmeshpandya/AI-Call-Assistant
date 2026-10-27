"""Transport boundary between the conversation engine and the client.

The engine only talks to `ClientTransport`. The browser WebSocket is one implementation
(api/websocket.py); a future telephony transport (SIP / provider media stream) implements the same
two methods and feeds 16 kHz mono PCM into `VoiceSession.handle_audio`.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.tts.tts_base import SynthesizedAudio


class ClientTransport(Protocol):
    async def send_event(self, event: dict[str, Any]) -> None:
        """Send a JSON-serializable control/event message."""

    async def send_audio(self, generation_id: int, seq: int, audio: SynthesizedAudio) -> None:
        """Send one synthesized sentence of assistant audio."""
