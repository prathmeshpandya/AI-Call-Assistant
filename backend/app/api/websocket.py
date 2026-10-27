"""Browser WebSocket transport: /ws

Inbound:  binary = PCM16 LE mono 16 kHz audio; text = JSON control message (api/protocol.py).
Outbound: text = JSON events; binary = [uint32 generation_id][uint32 seq][uint32 sample_rate] + PCM16 LE mono.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from app.api.protocol import parse_control
from app.audio.audio_utils import float32_to_pcm16, pack_audio_frame
from app.core.session_manager import SessionManager, TooManySessionsError
from app.tts.tts_base import SynthesizedAudio

log = logging.getLogger(__name__)
router = APIRouter()


class WebSocketTransport:
    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._lock = asyncio.Lock()  # several tasks send concurrently; frames must not interleave

    async def send_event(self, event: dict[str, Any]) -> None:
        async with self._lock:
            await self._ws.send_text(json.dumps(event))

    async def send_audio(self, generation_id: int, seq: int, audio: SynthesizedAudio) -> None:
        frame = pack_audio_frame(generation_id, seq, audio.sample_rate, float32_to_pcm16(audio.samples))
        async with self._lock:
            await self._ws.send_bytes(frame)


def _origin_allowed(ws: WebSocket, allowed: list[str]) -> bool:
    origin = ws.headers.get("origin")
    # Non-browser clients (tests, CLI tools) send no Origin; browsers always do.
    return origin is None or origin in allowed or "*" in allowed


@router.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    app = ws.app
    settings = app.state.settings
    manager: SessionManager = app.state.sessions
    if not _origin_allowed(ws, settings.allowed_origins):
        log.warning("Rejected WebSocket origin", extra={"origin": ws.headers.get("origin")})
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await ws.accept()
    transport = WebSocketTransport(ws)
    try:
        session = await manager.open(transport)
    except TooManySessionsError as exc:
        await transport.send_event({"type": "error", "code": "too_many_sessions", "message": str(exc), "stage": "session"})
        await ws.close(code=status.WS_1013_TRY_AGAIN_LATER)
        return

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if (data := message.get("bytes")) is not None:
                await session.handle_audio(data)
            elif (text := message.get("text")) is not None:
                try:
                    control = parse_control(text)
                except (ValidationError, ValueError) as exc:
                    await transport.send_event({"type": "error", "code": "bad_message", "stage": "protocol",
                                                "message": f"Invalid control message: {str(exc)[:200]}"})
                    continue
                await session.handle_control(control)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("WebSocket loop failed", extra={"session": session.id})
    finally:
        await manager.close(session.id)
